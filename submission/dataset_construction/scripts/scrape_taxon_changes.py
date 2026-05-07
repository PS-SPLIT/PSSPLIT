import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime

def get_taxonomic_splits(base_url, suffix, latest_date_str, earliest_date_str, save_name):
    latest_date = datetime.strptime(latest_date_str, '%d/%m/%Y')
    earliest_date = datetime.strptime(earliest_date_str, '%d/%m/%Y')
    all_splits = {}
    page=30

    while True:
        time.sleep(2)
        url = f"{base_url}{page}{suffix}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all split entries
        split_entries = soup.find_all('div', class_='taxon_change')
        if not split_entries:
            break  # Break if there are no entries on the page

        for entry in split_entries:
            split_id = entry.find('a', href=lambda x: x and x.startswith('/taxon_splits'))['href'].split('/')[-1]
            date_str = entry.find('h2').text.split('Committed on')[-1].strip().rstrip(')')
            date = datetime.strptime(date_str, '%Y-%m-%d')

            # Stop if the split is before or on the target date
            if date <= earliest_date:
                return all_splits

            # Don't add to the dataset if it is from after the latest date.
            if date >= latest_date:
                pass

            else:
                # Extracting source information
                source_section = entry.find('div', class_='meta')
                source_citation = source_section.find('div', class_='dialog')
                if source_citation and source_citation.find('a'):
                    source_text = source_citation.text.strip()
                    source_link = source_citation.find('a')['href']
                    source = f"{source_text}, link: {source_link}"
                else:
                    # Check if the source is in the text outside the citation
                    source_text = source_section.text.strip().split('\n')
                    source = next((s for s in source_text if 'Source:' not in s and 'Committed by' not in s), 'unknown')

                # Process taxa
                taxa_table = entry.find('table', class_='taxon-change-table')
                split_into_marker = taxa_table.find('td', class_='mergetype')

                # Source and generated species lists
                source_species_info = []
                generated_species_info = []

                # Process source species
                for taxon in split_into_marker.find_previous('td').find_all('div', class_='change_taxon clear'):
                    taxon_data = extract_taxon_data(taxon)
                    source_species_info.append(taxon_data)

                # Process generated species
                gen_ids = []
                for taxon in split_into_marker.find_next('td').find_all('div', class_='change_taxon clear'):
                    taxon_data = extract_taxon_data(taxon)
                    generated_species_info.append(taxon_data)
                    gen_ids.append(taxon_data['taxon_id'])


                all_splits[split_id] = {
                    'date': date.strftime('%Y-%m-%d'),
                    'source': source,
                    'source_species': source_species_info,
                    'generated_species': generated_species_info
                }

                print(f"Split ID: {split_id}, Species: {gen_ids}")

        # save data to JSON file
        with open(f'{save_name}', 'w') as file:
            json.dump(all_splits, file, indent=4)

        page += 1

    return all_splits

def extract_taxon_data(taxon):
    taxon_id = taxon.get('data-taxon-id')
    observations = taxon.find('span', class_='count').text if taxon.find('span', class_='count') else '0'
    status = 'Active' if taxon.find('span', class_='active_taxon_status') else 'Inactive'
    return {'taxon_id': taxon_id, 'observations': observations, 'status': status}

# Base URL for taxonomic splits
# base_url = " https://www.inaturalist.org/taxon_changes?filters%5Bancestor_taxon_id%5D=&filters%5Bancestor_taxon_name%5D=&filters%5Bchange_group%5D=&filters%5Bcommitted%5D=Yes&filters%5Bdrop%5D=0&filters%5Biconic_taxon_id%5D=&filters%5Bmerge%5D=0&filters%5Bsource_id%5D=&filters%5Bsplit%5D=1&filters%5Bstage%5D=0&filters%5Bswap%5D=0&filters%5Btaxon_id%5D=&filters%5Btaxon_name%5D=&filters%5Btaxon_scheme_id%5D=&filters%5Buser_id%5D=&page="
# base_url_suffix = "&utf8=%E2%9C%93"
base_url = "https://www.inaturalist.org/taxon_changes?utf8=%E2%9C%93&filters%5Bsplit%5D=0&filters%5Bsplit%5D=1&filters%5Bmerge%5D=0&filters%5Bswap%5D=0&filters%5Bstage%5D=0&filters%5Bdrop%5D=0&filters%5Bchange_group%5D=&filters%5Btaxon_scheme_id%5D=&filters%5Bcommitted%5D=Yes&filters%5Biconic_taxon_id%5D=&filters%5Btaxon_name%5D=&filters%5Btaxon_id%5D=&filters%5Bancestor_taxon_name%5D=&filters%5Bancestor_taxon_id%5D=&filters%5Bsource_id%5D=&page="
base_url_suffix = "&utf8=✓"
# target_date = "25/09/2020"
latest_date = "25/09/2020"
earliest_date = "01/01/2010"
save_name = 'splits_for_training_pre_25_09_2020.json'

# Scrape the first page
splits = get_taxonomic_splits(base_url, base_url_suffix, latest_date, earliest_date, save_name)

# Save data to JSON file
with open(save_name, 'w') as file:
    json.dump(splits, file, indent=4)

print(f"Data saved to {save_name}")