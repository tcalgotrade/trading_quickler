from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

"""
Setting up

Add directory of chromedriver/geckodriver and restart to get Selenium to recognize edited PATH
Open cmd prompt, cd C:\Program Files (x86)\Google\Chrome\Application or cd C:\Program Files \Google\Chrome\Application
Run chrome.exe --remote-debugging-port=9222
Opens an existing signed in browser: No need to relogin on sites, especially those with 2FA
"""

def press_asset_favorite():

    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://olymptrade.com/platform")

    # https://is.gd/DKtCfi ; 1st 2 works. Last one works on laptop, not RESTED.

    # Asset Class Button
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class,'ButtonBase-module-host-jU9 asset-button asset-button_on-platform')]"))).click()

    # List of favorites
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//span[contains(@data-trans,'assets_list_sort_fav')]"))).click()

    # Asset Info
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class,'ButtonBase-module-host-jU9 asset-button-info asset-item__button-info')]"))).click()

    # Asset
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//span[contains(@role,'btn slick-arrow slick-next slick-undefined btn_transparent btn_undefined') and contains(text(),'Quotes History')]"))).click()

    return
press_asset_favorite()

