"""Tests for database helper functions that don't require a live connection."""

from unittest.mock import MagicMock, patch

from features.database import clear_all_data, Device, Prediction, Alert, HistoricalData


class TestClearAllData:
    @patch("features.database.get_session")
    def test_deletes_devices_predictions_alerts_and_historical(self, mock_get_session):
        session = MagicMock()
        # each session.query(Model).delete() returns a row count
        session.query.return_value.delete.return_value = 7
        mock_get_session.return_value = session

        counts = clear_all_data()

        assert counts == {
            "historical_data": 7,
            "devices": 7,
            "predictions": 7,
            "alerts": 7,
        }
        # every targeted model was queried for deletion
        queried = {call.args[0] for call in session.query.call_args_list}
        assert {Device, Prediction, Alert, HistoricalData} <= queried
        session.commit.assert_called_once()
        session.close.assert_called_once()

    @patch("features.database.get_session")
    def test_historical_data_deleted_before_devices(self, mock_get_session):
        session = MagicMock()
        session.query.return_value.delete.return_value = 0
        mock_get_session.return_value = session

        clear_all_data()

        models_in_order = [call.args[0] for call in session.query.call_args_list]
        # FK safety: historical_data must be deleted before devices
        assert models_in_order.index(HistoricalData) < models_in_order.index(Device)

    @patch("features.database.get_session")
    def test_rolls_back_and_closes_on_error(self, mock_get_session):
        session = MagicMock()
        session.query.return_value.delete.side_effect = RuntimeError("boom")
        mock_get_session.return_value = session

        try:
            clear_all_data()
        except RuntimeError:
            pass

        session.rollback.assert_called_once()
        session.close.assert_called_once()
        session.commit.assert_not_called()
