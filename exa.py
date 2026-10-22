import os
import ast
import json
import sys
import re
import urllib.request
import urllib.error

# Core open-weights deep learning frameworks that are safe to leave as standard system imports
TRUSTED_FRAMEWORKS = {"torch", "torchvision", "numpy", "timm", "transformers", "safetensors"}

class ImportRefactorer(ast.NodeTransformer):
    """
    Walks an AST tree and converts module-scoped attribute calls to local flat calls.
    Example: foo.bar() -> bar()
    """
    def __init__(self, modules_to_strip):
        self.modules_to_strip = modules_to_strip

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.modules_to_strip:
            return ast.Name(id=node.attr, ctx=node.ctx)
        return self.generic_visit(node)

class InteractiveCodeExtractor:
    def __init__(self, main_repo_meta):
        self.main_repo = main_repo_meta
        self.global_imports = set()
        self.internalized_modules = set()
        self.fetched_nodes = {}

        # Build dynamic environment platform core whitelist
        self.system_whitelist = set(sys.builtin_module_names)
        if hasattr(sys, "stdlib_module_names"):
            self.system_whitelist.update(sys.stdlib_module_names)
        else:
            fallback_stdlib = {
                "os", "sys", "re", "math", "json", "collections", "typing", "functools",
                "abc", "time", "datetime", "urllib", "hashlib", "shutil", "ast", "inspect"
            }
            self.system_whitelist.update(fallback_stdlib)

        self.system_whitelist.update(TRUSTED_FRAMEWORKS)

    def _get_pypi_github_repo(self, package_name):
        url = f"https://pypi.org{package_name}/json"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
                info = data.get("info", {})
                project_urls = info.get("project_urls", {}) or {}
                search_pool = [info.get("home_page", "")] + list(project_urls.values())
                for link in search_pool:
                    if link and "github.com" in link:
                        match = re.search(r"github\.com/([^/]+)/([^/]+)", link)
                        if match:
                            user, repo = match.group(1), match.group(2)
                            repo = repo.split(".git")[0].split("/")[0]
                            return {"user": user, "repo": repo, "branch": "main"}
        except Exception:
            pass
        return None

    def _fetch_file_content(self, repo_meta, relative_path):
        # Determine if we are hitting Hugging Face or standard GitHub repos
        is_hf = repo_meta["repo"] == "DAT" or "w-e-w" in repo_meta["user"]

        branches = [repo_meta.get("branch", "main"), "main", "master"]
        last_url = ""

        for branch in branches:
            if is_hf:
                # Hugging Face serves raw text codes securely via /resolve/ endpoints
                url = f"https://huggingface.co/{repo_meta['user']}/{repo_meta['repo']}/resolve/{branch}/{relative_path}"
            else:
                url = f"https://raw.githubusercontent.com/{repo_meta['user']}/{repo_meta['repo']}/{branch}/{relative_path}"

            last_url = url
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    repo_meta["branch"] = branch
                    return response.read().decode('utf-8')
            except urllib.error.URLError:
                continue

        raise RuntimeError(f"Could not reach remote source code layout.\nAttempted URL: {last_url}")

    def _find_dependencies(self, node):
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
        return dependencies

    def extract_entity(self, repo_meta, file_path, entity_name):
        if entity_name in self.fetched_nodes:
            return

        print(f"➜ Scanning: '{entity_name}' inside {repo_meta['repo']}/{file_path}")
        code_str = self._fetch_file_content(repo_meta, file_path)
        tree = ast.parse(code_str)

        local_imports = {}

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split('.')[0]
                    self._handle_dependency_routing(root_pkg, alias.name + ".py", alias.name, local_imports, repo_meta)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root_pkg = module.split('.')[0]

                if node.level > 0:
                    base_dir = os.path.dirname(file_path)
                    module_path = os.path.normpath(os.path.join(base_dir, *module.split('.'))) + ".py"
                    for alias in node.names:
                        local_imports[alias.name] = (repo_meta, module_path, alias.name)
                else:
                    module_path = module.replace('.', '/') + ".py"
                    for alias in node.names:
                        self._handle_dependency_routing(root_pkg, module_path, alias.name, local_imports, repo_meta)

        target_node = None
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == entity_name:
                target_node = node
                break
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == entity_name:
                        target_node = node
                        break

        if not target_node:
            raise ValueError(f"Could not extract target component structural definition for: {entity_name}")

        deps = self._find_dependencies(target_node)
        for dep in deps:
            if dep in local_imports:
                next_repo, next_file, next_entity = local_imports[dep]
                try:
                    self.extract_entity(next_repo, next_file, next_entity)
                except Exception as e:
                    print(f"  ℹ️ Skipping external component mapping for '{dep}': {e}")
            elif dep != entity_name:
                for node in tree.body:
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == dep:
                        self.extract_entity(repo_meta, file_path, dep)

        refactorer = ImportRefactorer(self.internalized_modules)
        transformed_node = refactorer.visit(target_node)
        ast.fix_missing_locations(transformed_node)

        self.fetched_nodes[entity_name] = ast.unparse(transformed_node)

    def _handle_dependency_routing(self, root_pkg, module_path, entity_name, local_imports, current_repo):
        # Clean array lists to pure strings to match system whitelists seamlessly
        pkg_base_name = root_pkg if isinstance(root_pkg, str) else root_pkg[0]

        # AUTOMATED NATIVE SYSTEM INTERCEPT
        if pkg_base_name in self.system_whitelist:
            if pkg_base_name not in self.internalized_modules:
                self.global_imports.add(f"import {pkg_base_name}")
            return

        print(f"\n⚠️  [UNKNOWN DEPENDENCY DETECTED]: Found module reference '{pkg_base_name}' while resolving '{entity_name}'")
        choice = input(f"   Allow standard system runtime import for '{pkg_base_name}'? (y=whitelist / n=scrape source from GitHub): ").strip().lower()

        if choice == 'y':
            self.system_whitelist.add(pkg_base_name)
            self.global_imports.add(f"import {pkg_base_name}")
            return

        print(f"   Searching PyPI for '{pkg_base_name}' repository metadata...")
        remote_meta = self._get_pypi_github_repo(pkg_base_name)

        if not remote_meta:
            print(f"❌ Could not automatically locate a GitHub homepage for '{pkg_base_name}' via pip/PyPI registry metrics.")
            user_url = input(f"   Please paste the exact GitHub URL manually or press Enter to skip: ").strip()
            if user_url:
                match = re.search(r"github\.com/([^/]+)/([^/]+)", user_url)
                if match:
                    remote_meta = {"user": match.group(1), "repo": match.group(2).replace(".git",""), "branch": "main"}

        if remote_meta:
            print(f"   Successfully mapped source coordinates: {remote_meta['user']}/{remote_meta['repo']}")
            self.internalized_modules.add(pkg_base_name)
            local_imports[entity_name] = (remote_meta, module_path, entity_name)
        else:
            print(f"   Skipping automatic resolution hook for '{pkg_base_name}'. Treating as global dependency.")
            self.system_whitelist.add(pkg_base_name)
            self.global_imports.add(f"import {pkg_base_name}")

    def compile(self, output_file):
        with open(output_file, "w") as f:
            f.write("# ==========================================================\n")
            f.write(f"# COMPACT DEPLOYMENT MODULE - COMPLETELY FLAT NAMESPACE\n")
            f.write("# ==========================================================\n\n")

            for imp in sorted(list(self.global_imports)):
                f.write(f"{imp}\n")
            f.write("\n\n")

            for name, code in self.fetched_nodes.items():
                f.write(f"# --- Extracted definition for {name} ---\n")
                f.write(code)
                f.write("\n\n")
        print(f"\n🎉 Isolated deployment code generated cleanly at: '{output_file}'")

if __name__ == "__main__":
    target_project = {
        "user": "w-e-w",
        "repo": "DAT",
        "branch": "main"
    }

    extractor = InteractiveCodeExtractor(target_project)
    extractor.extract_entity(
        repo_meta=target_project,
        file_path="basicsr/archs/dat_arch.py",
        entity_name="DAT"
    )
    extractor.compile("production_print_scaler.py")


