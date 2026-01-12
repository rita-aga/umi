# Release Process for umi-memory

This document explains how to release a new version of `umi-memory` to crates.io.

## Automated Release Workflow

Publishing is **automated via GitHub Actions** when you push a version tag. Here's the complete process:

## Release Checklist

### 1. Prepare the Release

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Run tests locally to verify everything works
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo fmt --all
```

### 2. Update Version

Edit `Cargo.toml` (workspace version):

```toml
[workspace.package]
version = "0.2.0"  # Bump version here
```

### 3. Update CHANGELOG (Optional but Recommended)

Create or update `CHANGELOG.md` with notable changes:

```markdown
## [0.2.0] - 2026-01-XX

### Added
- New feature X
- New feature Y

### Fixed
- Bug Z

### Changed
- Breaking change: ...
```

### 4. Commit Version Bump

```bash
git add Cargo.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

### 5. Create and Push Tag

**This is what triggers the automated publish:**

```bash
# Create the tag locally
git tag v0.2.0

# Push the tag (THIS PUBLISHES TO CRATES.IO)
git push origin v0.2.0
```

### 6. Monitor the Release

The automated workflow will:

1. ✅ **Pre-Release Checks** (~5 min)
   - Verify Cargo.toml version matches tag
   - Run full test suite
   - Run clippy
   - Build documentation
   - Dry run publish

2. ✅ **Publish to crates.io** (~2 min)
   - Actual `cargo publish`
   - Wait for crates.io indexing

3. ✅ **Create GitHub Release** (~1 min)
   - Generate changelog from commits
   - Create release with notes
   - Link to crates.io and docs.rs

**Watch progress:** https://github.com/rita-aga/umi/actions

### 7. Verify Publication

After the workflow completes (~8 minutes):

- ✅ Check crates.io: https://crates.io/crates/umi-memory
- ✅ Check docs.rs: https://docs.rs/umi-memory
- ✅ Check GitHub release: https://github.com/rita-aga/umi/releases
- ✅ Test installation: `cargo add umi-memory@0.2.0`

## What If Something Goes Wrong?

### Version Mismatch Error

If Cargo.toml version doesn't match the tag:

```bash
# Delete the tag locally and remotely
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0

# Fix Cargo.toml
# Create tag again with correct version
```

### Publish Fails

If `cargo publish` fails:

1. Check the workflow logs for the error
2. Fix the issue in a new commit
3. Push the commit
4. Delete and recreate the tag:

```bash
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
git tag v0.2.0
git push origin v0.2.0
```

### Need to Yank a Version

If you published a broken version:

```bash
# Yank the version (doesn't delete, but prevents new uses)
cargo yank --vers 0.2.0 umi-memory

# Release a fixed version
# Bump to 0.2.1, commit, tag, push
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

**While in 0.x.x:**
- 0.x.0: Minor features, may have breaking changes
- 0.0.x: Bug fixes only

## Regular Development (No Publishing)

Normal commits and pushes **never trigger publishing**:

```bash
git add .
git commit -m "feat: add new feature"
git push origin main
```

→ CI runs tests, clippy, fmt
→ **Does NOT publish to crates.io**

Only pushing a tag triggers publishing!

## Summary

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Commit and push
4. Create tag: `git tag v0.2.0`
5. Push tag: `git push origin v0.2.0` ← **This publishes!**
6. Monitor workflow
7. Verify on crates.io, docs.rs, GitHub releases

That's it! The automation handles the rest.
