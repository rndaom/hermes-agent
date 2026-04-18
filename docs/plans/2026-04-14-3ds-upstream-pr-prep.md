# 3DS Upstream PR Prep

Date: 2026-04-14

## Goal

Capture the cleanup work that should happen before proposing the 3DS gateway/client integration upstream.

The current implementation is good for the fork: it preserves Hermes command names and behavior while adding a handheld-friendly UI and 3DS-specific gateway transport. Before an upstream PR, the 3DS-specific interaction flow should be generalized so it fits the rest of Hermes cleanly.

## Current State

The fork currently does the right thing in product terms:

- reuses real Hermes commands like `/model`, `/personality`, `/resume`, `/rollback`, `/reasoning`, and `/fast`
- reuses existing gateway concepts like `send_model_picker`, `send_exec_approval`, `reply_to`, `request_id`, and `choice`
- keeps the 3DS UI as a frontend for existing Hermes behavior rather than inventing parallel command semantics

The part that is not yet ideal for upstream is structural:

- `send_interaction_picker()` is currently a 3DS-specific adapter method
- `interaction.request` is currently a 3DS-specific event payload shape
- some picker logic in `gateway/run.py` is still explicitly branched for the 3DS path

## Upstream Prep Checklist

1. Promote picker interactions to a first-class generic adapter capability.
2. Add the capability to `BasePlatformAdapter` with a clear contract and fallback behavior.
3. Refactor 3DS, Telegram, and any future interactive adapters to conform to the same picker capability model.
4. Replace ad hoc 3DS checks in `gateway/run.py` with capability-based routing where practical.
5. Document the picker interaction lifecycle end-to-end:
   - request emission
   - option payload shape
   - interaction response semantics
   - cancel behavior
   - reply threading behavior
6. Review naming to ensure the generic API is Hermes-native and not handheld-specific.
7. Re-run focused gateway tests for approvals, model switching, reasoning, and 3DS adapter flows after the refactor.

## Preferred Refactor Shape

Suggested direction:

1. Define a generic adapter method for structured choice UIs.
2. Keep `send_exec_approval()` as the specialized approval helper when a platform wants a richer approval UX.
3. Make model/personality/resume/rollback flows call the generic picker path instead of a 3DS-only branch.
4. Treat the 3DS transport as one implementation of that generic picker system, not the source of the abstraction itself.

## Acceptance Criteria For A Future Upstream PR

The implementation is ready for upstream review when:

- picker interactions are described in shared adapter terms rather than 3DS-only terms
- `gateway/run.py` no longer needs special-case control flow just to support handheld choice UIs
- the 3DS-specific event payload is either generalized or clearly wrapped behind a generic platform interface
- gateway docs explain how interactive pickers relate to approvals and existing model pickers
- tests demonstrate that the generic picker flow works without changing command semantics

## Scope Boundary

This is not required for the fork to continue shipping.

It is only a prep note for the day we decide the 3DS work should be proposed to the Hermes maintainers.
