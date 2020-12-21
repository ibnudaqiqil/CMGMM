import re
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from requests import get
from string_utils.validation import is_integer

import utils


def author_scopus_docs(author_id, output_format='dictionary', pretty_print=None, xml_library='dicttoxml',
                       max_workers=None):
    url = f'http://sinta.ristekbrin.go.id/authors/detail?id={author_id}&view=documentsscopus'
    html = get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    page_info = soup.select('.uk-width-large-1-2.table-footer')
    n_page = utils.cast(page_info[0].text.strip().split()[3])
    worker_result = parse(soup,author_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page in range(2, n_page + 1):
            executor.submit(worker, author_id, page, worker_result)

    return utils.format_output(worker_result, output_format, pretty_print, xml_library)


def worker(author_id, page, worker_result):
    url = f'http://sinta.ristekbrin.go.id/authors/detail?page={page}&id={author_id}&view=documentsscopus'
    html = get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    data = parse(soup,author_id)

    worker_result.extend(data)


def parse(soup,author_id):
    rows = soup.select('table.uk-table tr')
    result = []

    for row in rows:
        link = row.select('a.paper-link')

        if not link:
            continue

        link = link[0]
        info1 = row.select('.index-val')
        quartile = info1[0].text.strip()
        citations = info1[1].text.strip()
        info2 = row.select('dd.indexed-by')[0].text.strip().split('|')

        result.append({
            'sinta_id':author_id,
            'title': link.text,
            'url': link['href'],
            'publisher': info2[0].strip(),
            'date': info2[3].strip(),
            'type': info2[4].strip(),
            'quartile': utils.cast(quartile[1]) if re.search(r'^Q[1-4]{1}$', quartile) else '-',
            'citations': utils.cast(citations) if is_integer(citations) else 0
        })

    return result

import pandas as pd
dosen_frame =pd.read_excel('datadosen.xls', index_col=0) 
scholar=[]
i=1
for index, row in dosen_frame[dosen_frame['scopus_document']>0].iterrows():
  print(row['sinta_id'])
  x_=author_scholar_docs2(row['sinta_id'])
  scholar.append(pd.DataFrame(x_))
scholar_list = (pd.concat(scholar))
model_save_name = 'datascopusdosen.xls'
scholar_list.to_excel(model_save_name)
