import requests
from bs4 import BeautifulSoup
import csv
from lxml import html

# Define the URL
url = "https://gptstore.ai/gpts?page=8"

# Fetch the webpage content
response = requests.get(url)
webpage = response.content

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(webpage, 'html.parser')

# Convert the soup object to a string and then to an lxml HTML element
dom = html.fromstring(str(soup))

# Open a CSV file to write the data
with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Title', 'Description'])

    # Use XPath to find the title and description elements
    title_elements = dom.xpath('//*[@id="__next"]/main/div[2]/ul/li/div/div[1]/div/a')
    description_elements = dom.xpath('//*[@id="__next"]/main/div[2]/ul/li/div/div[2]/div')

    # Iterate over the title elements and find corresponding descriptions
    for title_element, description_element in zip(title_elements, description_elements):
        title = title_element.text_content().strip() if title_element is not None else 'N/A'
        description = description_element.text_content().strip() if description_element is not None else 'N/A'
        
        # Print the title and description for debugging
        print("Title:", title)
        print("Description:", description)
        
        # Write the data to the CSV file
        writer.writerow([title, description])
