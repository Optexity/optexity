from optexity.inference.core.script_context import sanitize_download_filename


class TestSanitizeDownloadFilename:
    def test_replaces_hash_medical_records_style(self):
        assert (
            sanitize_download_filename("6_9_25 Prev. Medical Records #1_BgQRZ9B9Q2.pdf")
            == "6_9_25 Prev. Medical Records _1_BgQRZ9B9Q2.pdf"
        )

    def test_replaces_percent_and_ampersand(self):
        assert sanitize_download_filename("a%b&c.pdf") == "a_b_c.pdf"

    def test_preserves_spaces_and_unicode(self):
        assert sanitize_download_filename("报告 1.pdf") == "报告 1.pdf"

    def test_idempotent_for_already_safe_name(self):
        name = "6_9_25 Prev. Medical Records _1_BgQRZ9B9Q2.pdf"
        assert sanitize_download_filename(name) == name
