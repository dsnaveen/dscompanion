"""Tests for dscompanion/app/glossary.py — the static reference page explaining every
Step 5 feature-processing transformation.

The real regression value here is looping over every ``TRANSFORMER_REGISTRY``
key and actually rendering its live before/after example — this proves all 25
demos execute successfully against the real transformer classes (not just
that the page loads), and that a newly-registered transform can't silently
ship without a matching glossary entry (the module itself asserts this at
import time; this test exercises every entry end-to-end too).
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# dscompanion/app/ is a standalone script directory, not part of the installable
# dscompanion package (see streamlit.md) — add it to sys.path the same way
# streamlit_app.py does for its own sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from glossary import _CATEGORIES, _ENTRIES_BY_NAME, _NAMES_BY_CATEGORY

from dscompanion.features.registry import TRANSFORMER_REGISTRY


class TestGlossaryPage:
    def test_loads_with_no_dataset_or_prior_session_state(self):
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
        at.session_state["run_mode"] = "Glossary"
        at.run()
        assert not at.exception

    def test_every_registry_entry_has_a_glossary_entry(self):
        assert set(_ENTRIES_BY_NAME) == set(TRANSFORMER_REGISTRY)

    def test_every_transform_renders_a_live_before_after_example(self):
        for category in _CATEGORIES:
            for name in _NAMES_BY_CATEGORY[category]:
                at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
                at.session_state["run_mode"] = "Glossary"
                at.run()
                at.segmented_control(key="glossary.category").set_value(category).run()
                at.selectbox(key="glossary.transform").set_value(name).run()

                assert not at.exception, f"{name} raised an exception"
                assert not at.error, f"{name} rendered an error: {[e.value for e in at.error]}"
                dataframes = at.get("dataframe")
                assert len(dataframes) >= 2, f"{name} did not render before/after dataframes"
