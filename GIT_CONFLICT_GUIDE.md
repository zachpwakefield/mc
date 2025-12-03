# Resolving Git merge conflicts: incoming vs. current changes

When Git flags a conflict, it shows two versions of the code:
- **Current change** (sometimes shown as `<<<<<<< HEAD`): what is already in your branch.
- **Incoming change** (shown after `=======`): what you are trying to merge (e.g., main or another branch).

You choose one, the other, or a blend based on which version should survive in the final file. Use the context around the conflict (tests, recent features, or expected behavior) to decide.

## Quick decision checklist
1. **Understand the goal of each change.** Read both sides and the commit messages/PR descriptions if available.
2. **Prefer the code that matches the desired behavior.** If the incoming branch fixed a bug you also fixed differently, keep the best fix or combine them.
3. **Keep required updates.** If one side contains new API, schema, or config that other files rely on, keep that side or merge the relevant parts.
4. **Run tests after resolving.** This catches hidden breakages.

## How to resolve in practice
1. Open the conflicted file and locate markers:
   ```
   <<<<<<< HEAD
   current change
   =======
   incoming change
   >>>>>>> main
   ```
2. Decide whether to keep **current**, **incoming**, or a **manual merge**:
   - Keep current: delete the markers and incoming section.
   - Keep incoming: delete markers and current section.
   - Manual merge: edit to combine both (common when each side adds complementary lines).
3. Save the file without the conflict markers.
4. Mark resolved and commit:
   ```bash
   git add <file>
   git commit
   ```

## Tips for choosing correctly
- **Check for dependencies:** If a related file expects new names/fields introduced on the incoming side, keep those updates.
- **Avoid losing fixes:** Search the history (`git blame`, PR notes) to ensure you keep bug fixes or important refactors.
- **Prefer clarity:** If both sides are valid, merge them into the clearest combined version.
- **If unsure, test both:** Temporarily keep one side, run tests; if failing, try the other or combine.

## See also
- `git checkout --theirs <file>` keeps the incoming version.
- `git checkout --ours <file>` keeps the current version.
- GUI tools (`git mergetool`, VS Code, etc.) can show side-by-side views.

