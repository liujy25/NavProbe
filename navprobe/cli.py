"""Command entry points for VLNCE, HM3D and Habitat installation checks."""

from __future__ import annotations


def vlnce_main() -> None:
    from navprobe.runners.vlnce_runner import main

    main()


def hm3d_batch_main() -> None:
    from navprobe.runners.hm3d_batch_runner import main

    main()


def habitat_smoke_main() -> None:
    from navprobe.env.habitat_smoke import main

    main()
