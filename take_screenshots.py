import os
from playwright.sync_api import sync_playwright

os.makedirs("screenshots", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900},
                            device_scale_factor=2)
    page.goto("http://127.0.0.1:5000")

    for term, label in [("itching", "Itching"),
                        ("skin rash", "Skin Rash"),
                        ("nodal skin", "Nodal Skin Eruptions")]:
        page.fill("#symptomSearch", term)
        page.click(f"#suggestions button:text-is('{label}')")

    page.evaluate("document.activeElement.blur()")
    page.screenshot(path="screenshots/home.png")

    page.click("#submitBtn")
    page.wait_for_url("**/predict")
    page.screenshot(path="screenshots/results.png", full_page=True)

    browser.close()

print("Saved screenshots/home.png and screenshots/results.png")