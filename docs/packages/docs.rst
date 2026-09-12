dscompanion.docs
=============

:class:`dscompanion.ModelCard` is documented in the :doc:`top-level API reference
<../api/dscompanion>` (it's the sub-package's only public class, and is re-exported at the
top level).

``ModelCard`` is the governance-ready report — the output of pipeline stage 13.
``.generate()`` introspects the model, split, and every optional collaborator
(``explainer``, ``calibrator``, ``eda_report``, ``tuner``, ``leaderboard``) into 15
sections (``_SECTION_KEYS``: config summary, model summary, data lineage, EDA summary,
feature inventory, performance metrics, decile table, stability, calibration,
explainability, hyperparameter tuning, leaderboard, limitations, governance, full
config). Renders via four independent methods:

- ``to_html(path)`` — Jinja2 template (``"generic"`` or ``"credit_risk"``), self-contained
  (Bootstrap + Plotly inlined).
- ``to_word(path)`` — ``.docx``, one heading + table per section, generic (no per-section
  custom rendering).
- ``to_excel(path)`` — one worksheet per section plus a navigable ``Index`` sheet and
  native xlsxwriter charts (no Plotly/kaleido image embedding). This is the only format
  the leaderboard section currently renders as a visual ranked table — ``to_html()`` /
  ``to_word()`` only show ``leaderboard.enabled`` in the config summary and the raw
  config dump.
- ``to_dict()`` — JSON-serialisable, DataFrame columns converted to record lists.
