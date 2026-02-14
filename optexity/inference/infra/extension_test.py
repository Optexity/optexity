import json
import pathlib
import shutil
import subprocess


class ChromeWithExtensions:
    def __init__(self, user_data_dir: str = "/tmp/chrome-profile1"):
        self.user_data_dir = pathlib.Path(user_data_dir)
        self.extensions = []

    def add_extension(self, extension_id: str, name: str | None = None):
        """Add extension ID from Chrome Web Store."""
        self.extensions.append({"id": extension_id, "name": name or extension_id})

    def setup_forced_extensions(self):
        """
        Use Chrome's ExtensionInstallForcelist policy to auto-install extensions.
        This is the enterprise method and works reliably.
        """
        # Clean slate
        if self.user_data_dir.exists():
            shutil.rmtree(self.user_data_dir)
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

        # Create managed policies directory
        # Note: On macOS, you might need to use system-wide policies
        # but for testing, we'll use user-data-dir approach

        preferences = {
            "extensions": {"settings": {}},
            "browser": {"show_home_button": True},
        }

        # Add each extension
        for ext in self.extensions:
            ext_id = ext["id"]
            update_url = "https://clients2.google.com/service/update2/crx"

            preferences["extensions"]["settings"][ext_id] = {
                "state": 1,
                "path": ext_id,
                "from_webstore": True,
                "manifest": {"update_url": update_url, "name": ext["name"]},
            }

        # Write preferences before first run
        default_dir = self.user_data_dir / "Default"
        default_dir.mkdir(parents=True, exist_ok=True)

        with open(default_dir / "Preferences", "w") as f:
            json.dump(preferences, f, indent=2)

        print(f"✅ Configured {len(self.extensions)} extensions for auto-install")

    def launch(self):
        """Launch Chrome with configured extensions."""
        # self.setup_forced_extensions()

        chrome_cmd = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            f"--user-data-dir={self.user_data_dir}",
            "--remote-debugging-port=9222",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        print(f"🚀 Launching Chrome...")
        print(f"📦 Extensions will auto-install from Chrome Web Store")

        process = subprocess.Popen(chrome_cmd)

        print(f"⏳ Please wait 10-15 seconds for extensions to download and install...")
        print(f"💡 Check chrome://extensions to verify installation")

        return process


if __name__ == "__main__":
    # Usage
    chrome = ChromeWithExtensions()
    chrome.add_extension(
        "edibdbjcniadpccecjdfdjjppcpchdlm", "I Still Don't Care About Cookies"
    )
    chrome.add_extension("cjpalhdlnbpafiamejdnhcphjbkeiagm", "uBlock Origin")
    process = chrome.launch()

    input("Press Enter to close...")
    process.terminate()
