from uuid import uuid4
import argparse
import logging
import pathlib
import hashlib
import time
import json
import sys
import os

import pathspec
import redis as RedisLib

# Additional file extensions to skip
SKIP_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.bin',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
    '.mp3', '.mp4', '.avi', '.mov', '.mkv',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.min.js', '.min.css',  # Minified files
    '.map',  # Source maps
    '.lock',  # Lock files
}


class GitignoreMatcher:
    """Handles gitignore pattern matching across a directory tree."""
    
    def __init__(self, root_path):
        self.root_path = pathlib.Path(root_path).resolve()
        self.specs = []
        self.logger = logging.getLogger(f"{__name__}.GitignoreMatcher")
        self._load_default_ignores()
        self._load_gitignore_files()
        
    def _load_default_ignores(self):
        """Load default ignore patterns from JSON file."""
        default_ignores_path = pathlib.Path(__file__).parent / "default-ignores.json"
        
        default_patterns = []
        
        try:
            if default_ignores_path.exists():
                self.logger.info(f"Loading default ignores from {default_ignores_path}")
                with open(default_ignores_path, 'r', encoding='utf-8') as f:
                    ignore_data = json.load(f)
                    
                # Flatten all patterns from all categories
                for category, patterns in ignore_data.items():
                    default_patterns.extend(patterns)
                    self.logger.debug(f"Loaded {len(patterns)} patterns from category '{category}'")
                    
                self.logger.info(f"Loaded {len(default_patterns)} default ignore patterns")
            else:
                self.logger.warning(f"Default ignores file not found at {default_ignores_path}")
                # Fallback to minimal set if file not found
                default_patterns = [
                    '.git/',
                    '__pycache__/',
                    '*.pyc',
                    'node_modules/',
                    '.venv/',
                    'venv/',
                ]
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing default-ignores.json: {e}")
            # Fallback patterns
            default_patterns = ['.git/', '__pycache__/', '*.pyc']
        except Exception as e:
            self.logger.error(f"Error loading default ignores: {e}")
            default_patterns = ['.git/', '__pycache__/', '*.pyc']
        
        # Create default spec
        if default_patterns:
            default_spec = pathspec.PathSpec.from_lines('gitwildmatch', default_patterns)
            self.specs.append((self.root_path, default_spec))
        
    def _load_gitignore_files(self):
        """Load all .gitignore files in the directory tree."""
        # Load all .gitignore files
        gitignore_count = 0
        for gitignore_path in self.root_path.rglob('.gitignore'):
            try:
                self.logger.info(f"Loading gitignore: {gitignore_path}")
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    patterns = []
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            patterns.append(line)
                    
                    if patterns:
                        spec = pathspec.PathSpec.from_lines('gitwildmatch', patterns)
                        base_dir = gitignore_path.parent.resolve()
                        self.specs.append((base_dir, spec))
                        self.logger.info(f"Loaded {len(patterns)} patterns from {gitignore_path}")
                        gitignore_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error loading {gitignore_path}: {e}")
        
        self.logger.info(f"Loaded {gitignore_count} gitignore files, total specs: {len(self.specs)}")
    
    def should_ignore(self, file_path):
        """Check if a file should be ignored based on all gitignore rules."""
        file_path = pathlib.Path(file_path).resolve()
        
        # Check each gitignore spec
        for base_dir, spec in self.specs:
            try:
                # Get the relative path from the gitignore's directory
                if file_path.is_relative_to(base_dir):
                    rel_path = file_path.relative_to(base_dir)
                    
                    # Check if this path matches any pattern
                    if spec.match_file(str(rel_path)):
                        self.logger.debug(f"File {file_path} matched gitignore in {base_dir}")
                        return True
                    
                    # Also check with leading slash (for absolute patterns in gitignore)
                    if spec.match_file('/' + str(rel_path)):
                        self.logger.debug(f"File {file_path} matched absolute pattern in {base_dir}")
                        return True
                        
            except ValueError:
                # file_path is not under base_dir, skip this spec
                continue
            except Exception as e:
                self.logger.warning(f"Error checking {file_path} against {base_dir}: {e}")
                
        return False


def get_default_ignores():
    """Get a list of all default ignore patterns for display/debugging."""
    default_ignores_path = pathlib.Path(__file__).parent / "default-ignores.json"
    
    try:
        with open(default_ignores_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


if __name__ == '__main__':
    """The purpose of this init py is to allow us to 
    run this rag
    """
    REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
    redis = RedisLib.Redis(host=REDIS_HOST, decode_responses=True)
    
    runid = uuid4().hex
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="/code")
    p.add_argument("--force-reload", default=False, action='store_true')
    p.add_argument("--runid", type=str, default=runid)
    p.add_argument("--show-default-ignores", action='store_true', help="Show default ignore patterns and exit")
    args = p.parse_args()
    
    # Show default ignores if requested
    if args.show_default_ignores:
        print("Default ignore patterns:")
        ignores = get_default_ignores()
        for category, patterns in ignores.items():
            print(f"\n{category}:")
            for pattern in patterns:
                print(f"  - {pattern}")
        sys.exit(0)
    
    runid = args.runid
    log = logging.getLogger(__file__ + "-" + runid)
    logging.basicConfig(
        level=logging.INFO,
        format=f"rag background task({runid=}) | %(filename)s - %(lineno)d %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    
    from .. import CodeDocument, CodeChunk
    from . import es
    from . import tokenizers
    from . import ollama_util
    
    code_path = pathlib.Path(args.directory).resolve()
    assert code_path.exists()
    
    # Create the gitignore matcher
    log.info("Initializing gitignore matcher...")
    gitignore_matcher = GitignoreMatcher(code_path)
    
    # Bulk indexer should autoflush
    processed_files = {}  # Map file path to hash
    start_time = time.time()
    file_count = 0
    skipped_count = 0
    ignored_count = 0
    
    with es.BulkIndexer(batch_size=5_000) as bulk:
        for loc in code_path.rglob("*"):
            # Skip directories
            if loc.is_dir():
                continue
            
            # Skip files with certain extensions
            if loc.suffix.lower() in SKIP_EXTENSIONS:
                log.debug(f"Skipping file with excluded extension: {loc}")
                skipped_count += 1
                continue
            
            # Skip large files (> 5MB)
            try:
                file_size = loc.stat().st_size
                if file_size > 5e6:
                    log.debug(f"Skipping large file ({file_size/1e6:.1f}MB): {loc}")
                    skipped_count += 1
                    continue
                elif file_size == 0:
                    log.debug(f"Skipping empty file: {loc}")
                    skipped_count += 1
                    continue
            except OSError as e:
                log.warning(f"Cannot stat file {loc}: {e}")
                continue
            
            # Check if file should be ignored
            if gitignore_matcher.should_ignore(loc):
                log.debug(f"Ignoring file per gitignore rules: {loc}")
                ignored_count += 1
                continue
            
            file_t = time.time()
            
            # Calculate file hash
            try:
                buf = hashlib.sha512()
                with loc.open('rb') as f:
                    # Read in chunks for large files
                    while chunk := f.read(8192):
                        buf.update(chunk)
                
                h = buf.hexdigest()
            except Exception as e:
                log.error(f"Error reading file {loc}: {e}")
                continue
            
            # Create Redis key that includes relative file path
            try:
                rel_path = loc.relative_to(code_path)
            except ValueError:
                log.error(f"File {loc} is not under code path {code_path}")
                continue
                
            redis_key = f"hash/{rel_path}/{h}"
            
            # Check if file was already processed
            cached_timestamp = redis.get(redis_key)
            
            should_index = (
                args.force_reload or 
                not cached_timestamp or 
                (time.time() - float(cached_timestamp)) > 24 * 60 * 60  # 24 hours
            )
            
            if should_index:
                log.info(
                    f"Indexing {'new' if not cached_timestamp else 'expired'} file: {rel_path}"
                )
                
                try:
                    token_count = 0
                    for token in tokenizers.tokenize(loc):
                        vec = ollama_util.get_embeddings(token.content)
                        doc = CodeDocument(
                            **token.model_dump(),
                            embedding=vec,
                        )
                        bulk.add(doc)
                        token_count += 1
                    
                    # Update cache after successful indexing
                    processed_files[redis_key] = time.time()
                    file_count += 1
                    log.info(f"Indexed {token_count} tokens from {rel_path}")
                    
                except Exception as e:
                    log.error(f"Error processing file {loc}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                log.debug(f"Skipping cached file: {rel_path}")
            
            log.debug(f"Processed {loc} in {time.time() - file_t:,.4f}s")
    
    # Update Redis with current timestamps for successfully processed files
    for redis_key, timestamp in processed_files.items():
        redis.set(redis_key, str(timestamp))
    
    elapsed = time.time() - start_time
    log.info(
        f"Indexing complete: indexed {file_count} files, "
        f"ignored {ignored_count} files (gitignore), "
        f"skipped {skipped_count} files (size/type) "
        f"in {elapsed:,.2f}s"
    )