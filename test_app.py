import os
import unittest
from unittest.mock import patch, MagicMock
import app

class TestApp(unittest.TestCase):
    def setUp(self):
        # Mock streamlit functions to avoid errors when running tests
        self.st_patcher = patch('app.st')
        self.mock_st = self.st_patcher.start()

    def tearDown(self):
        self.st_patcher.stop()
        # Clean up the downloaded model file if it exists
        if os.path.exists(app.MODEL_PATH):
            os.remove(app.MODEL_PATH)

    @patch('app.requests.Session')
    @patch('app.load_model')
    def test_download_model_success(self, mock_load_model, mock_session):
        """
        Test that the model is downloaded successfully by mocking the download.
        """
        # We patch load_model to prevent it from actually loading the model
        mock_load_model.return_value = "mock_model"

        # Configure the mock session and response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'dummy model data']

        mock_session_instance = mock_session.return_value
        mock_session_instance.get.return_value = mock_response

        # Call the download function
        model = app.download_model()

        # Check that the model file was created
        self.assertTrue(os.path.exists(app.MODEL_PATH))
        # Check that the file is not empty
        self.assertGreater(os.path.getsize(app.MODEL_PATH), 0)
        # Check that the function returns the mocked model
        self.assertEqual(model, "mock_model")
        # Check that success message was called
        self.mock_st.success.assert_called_with("Model downloaded successfully!")

if __name__ == '__main__':
    unittest.main()