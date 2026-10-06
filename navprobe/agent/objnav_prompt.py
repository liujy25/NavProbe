OBJNAV_TASK_CONSTRAINTS = """
Indoor ObjectNav task requirements:
- Target: a real instance inside the building. An outdoor object, reflection in glass or a mirror, photograph, or screen depiction does not satisfy the task.
- Approach: reach the target along connected navigable indoor floor. An indoor target seen through glass is a search cue; find an open route around the barrier. Glass, closed doors, walls, and enclosures are not shortcuts.
- Stop: physically occupy navigable floor close to the target in the same accessible local region, with no intervening barrier. Visibility, recognition, apparent image proximity, or closeness on the other side of glass does not establish this relation.
- If target validity, connected access, or the stopping relation is unconfirmed, keep the task incomplete and continue observing or searching for an accessible indoor instance.
""".strip()
