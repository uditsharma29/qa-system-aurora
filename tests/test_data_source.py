import unittest
from unittest import mock

import requests

from app import data_source


class DataSourceTests(unittest.TestCase):
    @mock.patch("app.data_source._fetch_page")
    def test_get_messages_paginates_until_total(self, mock_fetch_page):
        mock_fetch_page.side_effect = [
            (
                [
                    {"user_name": "Layla", "message": "One"},
                    {"user_name": "Vikram", "message": "Two"},
                ],
                3,
            ),
            (
                [
                    {"user_name": "Layla", "message": "Three"},
                ],
                3,
            ),
        ]

        records = data_source.get_messages()

        self.assertEqual(len(records), 3)
        self.assertEqual(mock_fetch_page.call_args_list[0].args[0], 0)
        self.assertEqual(mock_fetch_page.call_args_list[1].args[0], 2)

    @mock.patch("app.data_source._fetch_page")
    def test_get_messages_raises_on_unknown_http_error(self, mock_fetch_page):
        response = mock.Mock(status_code=403)
        response.status_code = None
        mock_fetch_page.side_effect = requests.exceptions.HTTPError(response=response)

        with self.assertRaises(requests.exceptions.HTTPError):
            data_source.get_messages()


if __name__ == "__main__":
    unittest.main()


