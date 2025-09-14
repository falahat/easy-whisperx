"""Basic functionality tests for easy_whisperx."""
import unittest


class TestStringMethods(unittest.TestCase):
    """Test basic string operations and imports."""

    def test_upper(self):
        """Test string upper() method."""
        self.assertEqual("foo".upper(), "FOO")

    def test_isupper(self):
        """Test string isupper() method."""
        self.assertTrue("FOO".isupper())
        self.assertFalse("Foo".isupper())

    def test_split(self):
        """Test string split() method."""
        s = "hello world"
        self.assertEqual(s.split(), ["hello", "world"])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()
