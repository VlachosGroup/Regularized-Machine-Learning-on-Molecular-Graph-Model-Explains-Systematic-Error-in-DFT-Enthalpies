# -*- coding: utf-8 -*-
"""
USed to scrape enthalpies from NIST database.
"""
from argparse import ArgumentParser

from lxml import html
from lxml import etree
import requests
import sys
import pickle


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output_folder',
                        help='Folder to save downloaded files')
    args = parser.parse_args()
    TIMEOUT = 100
    fin = 'NIST-CHONP.txt'
    fout = 'NIST-CHONP.pickle'
    urls = 'http://webbook.nist.gov/cgi/cbook.cgi?'
    urle = '&Units=CAL&cTG=on'

    fldrpath = args.output_folder
    fobjin = open(fldrpath + fin, 'r')
    lines = fobjin.readlines()
    fobjin.close()
    fobjout = open(fldrpath + fout, 'wb')
    data = list()
    imax = len(lines)
    for i in range(0, imax):
        # progress
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d/%d %d%%" % ('=' * int((i * 20 / imax + 1)), 
                         i, imax, 
                         i * 100 / imax + 1))
        sys.stdout.flush()
        # Read line
        line = lines[i]
        line = line.replace('\n', '')
        line = line.split('\t')
        # Reading data
        tryname = False
        # Try CAS first
        if line[6] != 'N/A':
            url = urls + 'ID=' + line[6] + urle
            page = requests.get(url)
            tree = html.fromstring(page.content)
            # check if the page exist
            title = tree.xpath('//meta[@content="Registry Number Not Found"]')
            if title:
                tryname = True
        else:
            tryname = True
        # Try Name url
        if tryname:
            line[0] = line[0].replace('&#945;', 'α')
            line[0] = line[0].replace('&#946;', 'β')
            line[0] = line[0].replace('&#947;', 'γ')
            line[0] = line[0].replace('&#948;', 'δ')
            url = urls + 'Name=' + line[0] + urle
            page = requests.get(url, timeout = TIMEOUT)
            tree = html.fromstring(page.content)
            # check if the page exist
            title = tree.xpath('//meta[@content="Name Not Found"]')
            if title:
                line.append('WebpageLoadingFailed')
                data.append(line)
                continue

        # Initialize variables
        sf_url = list()  # structure file url. [2d, 3d]. False if missing
        dh = list()  # heat of formation [value, CI]. CI is False, if missing
        siso = list()  # stereoisomers name and url
        # Get dH. If nothing available, dh is a empty list
        prop_table = tree.xpath('//table[@aria-label="One dimensional data"]')
        if prop_table:  # if there is prob
            prop_names = prop_table[0].xpath('.//td[1]')
            props = prop_table[0].xpath('.//td[2]')

            for i in range(0, len(prop_names)):
                if prop_names[i].text_content() == u'fH\xb0gas':
                    prop = props[i].text.split()
                    if prop:  # sometimes this value is empty. See Cubane
                        if len(prop) == 3:
                            dh.append([float(prop[0]), float(prop[2])])
                        else:
                            dh.append([float(prop[0]), False])
        if dh:
            line.append('Good')
        else:
            line.append('DHfMissing')
            data.append(line)
            continue

        # Get 2d mol and 3dmol data. If nothing available,
        # it returns a empty list
        structure = tree.xpath('//li[contains(.,"Chemical structure:")]')
        if structure:  # if there is structure
            f2d = structure[0].xpath('//a[text()="2d Mol file"]')
            f3d = structure[0].xpath('//a[text()="3d SD file"]')
            if f2d:
                sf_url.append('http://webbook.nist.gov' + f2d[0].values()[0])
            else:
                sf_url.append([False])
            if f3d:
                sf_url.append('http://webbook.nist.gov' + f3d[0].values()[0])
            else:
                sf_url.append([False])

        # Get stereoisomers. If nothing available, it returns a empty list
        stereoisomers = tree.xpath('//li[contains(.,"Stereoisomers:")]')
        if stereoisomers:  # if there is structure
            stereoisomers = stereoisomers[0].xpath('.//a')
            for isomer in stereoisomers:
                name = isomer.text_content()
                if isinstance(name, etree._ElementUnicodeResult):
                  name = name
                siso.append([name, 'http://webbook.nist.gov' + isomer.values()[0]])
        line.append([dh, sf_url, siso])
        data.append(line)
    pickle.dump(data, fobjout)
    fobjout.close()

