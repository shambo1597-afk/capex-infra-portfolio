"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def no_corporate_action_downloads(monkeypatch):
    """Unit tests never call NSE: price adjustment sees 'no corporate actions' unless a test
    passes its own actions to corporate_actions.adjust_for_corporate_actions()."""
    monkeypatch.setattr("corporate_actions.fetch_corporate_actions", lambda *args, **kwargs: [])
