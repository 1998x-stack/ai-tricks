"""离线构建全站并检查内容覆盖、子路径、链接及中文锚点。"""
import contextlib
import io
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from tools.build_site import Site, ROOT


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.h1, self.language = set(), [], 0, None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'h1':
            self.h1 += 1
        if tag == 'html':
            self.language = attrs.get('lang')
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


class StaticSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.site = Site(cls.root, '/ai-tricks')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.site.build()
        cls.documents = {p: Document(p.read_text()) for p in cls.root.rglob('*.html')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_internal_file_and_anchor_resolves(self):
        errors = []
        for page, document in self.documents.items():
            for link in document.links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                if url.path.startswith('/ai-tricks/'):
                    target = self.root / unquote(url.path).removeprefix('/ai-tricks/')
                else:
                    target = page.parent / unquote(url.path) if url.path else page
                if target.is_dir():
                    target /= 'index.html'
                target = target.resolve()
                if not target.exists():
                    errors.append((page.name, link, 'file missing'))
                elif url.fragment and target in self.documents:
                    if unquote(url.fragment) not in self.documents[target].ids:
                        errors.append((page.name, link, 'anchor missing'))
        self.assertEqual(errors, [])

    def test_every_wiki_and_protocol_has_readable_html_and_search_record(self):
        index = json.loads((self.root/'assets/search-index.json').read_text())
        urls = {item['url'] for item in index}
        self.assertEqual(len(urls), len(index))
        for source, route in self.site.pages.items():
            if '?' not in route:
                self.assertTrue((self.root/route).is_file(), source)
                self.assertIn(self.site.url(route), urls)
        self.assertEqual(len(list((self.root/'wiki').rglob('*.html'))), 72)
        self.assertEqual(len(list((self.root/'experiments').glob('*/index.html'))), 517)

    def test_semantic_chinese_pages_and_honest_evidence_status(self):
        for page, document in self.documents.items():
            self.assertEqual(document.language, 'zh-CN', page)
            self.assertEqual(document.h1, 1, page)
            self.assertIn('main', document.ids)
        home = (self.root/'index.html').read_text()
        self.assertIn('真实训练完成：0', home)
        self.assertIn('协议示意 · 非实验结果', home)
        manifest = json.loads((self.root/'build-manifest.json').read_text())
        self.assertEqual(manifest['html_pages'], len(self.documents))

    def test_formulas_are_native_mathml_without_javascript(self):
        body = (self.root/'wiki/llm-sft/lora.html').read_text()
        self.assertIn('<math ', body)
        self.assertIn('<mfrac>', body)
        self.assertNotIn('MATHSITETOKEN', body)
        self.assertNotIn('$$', body)

    def test_root_and_project_urls(self):
        self.assertEqual(self.site.url('experiments/index.html'), '/ai-tricks/experiments/')
        self.assertEqual(self.site.url('experiments/index.html?category=nlp'), '/ai-tricks/experiments/?category=nlp')
        root = Site(self.root, '')
        self.assertEqual(root.url('search/index.html'), '/search/')
        self.assertEqual(root.resolve('he-init.md', ROOT/'wiki/initialization/initialization.md'), '/wiki/initialization/he-init.html')

    def test_wikilinks_in_code_remain_literal_and_missing_links_are_explained(self):
        path = self.root/'fixture.md'
        path.write_text('# Fixture\n\n[[missing|待建]]\n\n```text\n[[leave-code-alone]]\n```\n')
        _, body, _, _ = self.site.render(path)
        self.assertIn('class="unresolved"', body)
        self.assertIn('[[leave-code-alone]]', body)
        self.assertNotIn('href="missing', body)


if __name__ == '__main__':
    unittest.main()
