"""Validate distributable archives, complete coverage, navigation and offline formulas."""
import json
import posixpath
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def test_xmind_complete_unique_topics_and_cross_sheet_links(self):
        from tools.build_exports import identity
        with zipfile.ZipFile(ROOT/'exports/AI-Tricks-中文思维导图.xmind') as package:
            self.assertIsNone(package.testzip())
            sheets = json.loads(package.read('content.json'))
            self.assertEqual(len(sheets), 15)
            manifest = json.loads(package.read('manifest.json'))
            self.assertTrue(set(manifest['file-entries']) <= set(package.namelist()))
        ids, hrefs = [], []
        def walk(topic):
            ids.append(topic['id'])
            self.assertTrue(topic['title'])
            if 'href' in topic:
                hrefs.append(topic['href'])
            for child in topic.get('children', {}).get('attached', []):
                walk(child)
        for sheet in sheets:
            walk(sheet['rootTopic'])
        self.assertEqual(len(ids), len(set(ids)))
        for href in hrefs:
            if href.startswith('xmind:#'):
                self.assertIn(href.removeprefix('xmind:#'), ids)
        for catalog in ('tricks', 'experiments'):
            for item in json.loads((ROOT/f'experiments/catalog/{catalog}.json').read_text()):
                self.assertIn(identity(item['id']), ids)

    def test_epub_package_navigation_and_every_internal_anchor(self):
        with zipfile.ZipFile(ROOT/'exports/AI-Tricks-中文实验手册.epub') as package:
            self.assertIsNone(package.testzip())
            first = package.infolist()[0]
            self.assertEqual(first.filename, 'mimetype')
            self.assertEqual(first.compress_type, zipfile.ZIP_STORED)
            self.assertEqual(package.read('mimetype'), b'application/epub+zip')
            xmls = {name: ET.fromstring(package.read(name)) for name in package.namelist()
                    if name.endswith(('.xhtml', '.opf', '.ncx', '.xml', '.svg'))}
            names = set(package.namelist())
            anchors = {name: {node.get('id') for node in root.iter() if node.get('id')}
                       for name, root in xmls.items()}
            for name, root in xmls.items():
                for node in root.iter():
                    for attr in ('href', 'src'):
                        href = node.get(attr)
                        if not href:
                            continue
                        url = urlsplit(href)
                        if url.scheme or url.netloc:
                            continue
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path))) if url.path else name
                        self.assertIn(target, names, (name, href))
                        if url.fragment:
                            self.assertIn(unquote(url.fragment), anchors[target], (name, href))
            opf = next(root for name, root in xmls.items() if name.endswith('.opf'))
            self.assertEqual(opf.find('.//{http://purl.org/dc/elements/1.1/}language').text, 'zh-CN')
            nav = next(item for item in opf.findall('.//{http://www.idpf.org/2007/opf}item') if item.get('properties') == 'nav')
            self.assertIsNotNone(nav)

    def test_epub_has_all_documents_and_preserves_116_native_formulas(self):
        from tools.build_exports import identity
        from tools.build_site import Site
        site = Site(ROOT/'exports', '/ai-tricks')
        with zipfile.ZipFile(ROOT/'exports/AI-Tricks-中文实验手册.epub') as package:
            texts = [package.read(name).decode() for name in package.namelist() if name.endswith('.xhtml')]
        joined = '\n'.join(texts)
        for source, route in site.pages.items():
            if '?' not in route:
                self.assertIn('id="'+identity(str(source.relative_to(ROOT)))+'"', joined)
        self.assertNotIn('XMATHMLTOKEN', joined)
        self.assertNotIn('href="/ai-tricks/', joined)
        formulas = sum(len(list(ET.fromstring(text).iter('{http://www.w3.org/1998/Math/MathML}math'))) for text in texts)
        self.assertEqual(formulas, 116)


if __name__ == '__main__':
    unittest.main()
