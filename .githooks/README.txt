Run once per clone (repo root):

  git config core.hooksPath .githooks

prepare-commit-msg and commit-msg strip Cursor co-author lines (cursoragent@cursor.com
and Co-authored-by: Cursor). Your git history keeps only you as author.

If you still see that line in Cursor's commit box, it is often stale text in
.git/COMMIT_EDITMSG — run:

  sed -i '' '/cursoragent/d;/Co-authored-by: Cursor/d' .git/COMMIT_EDITMSG

Or commit from the terminal: git commit -m "your message"
