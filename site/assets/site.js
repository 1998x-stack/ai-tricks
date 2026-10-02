'use strict';
const base = document.documentElement.dataset.base || '';
const navToggle = document.querySelector('.menu-toggle');
navToggle?.addEventListener('click', () => {
  const open = navToggle.getAttribute('aria-expanded') !== 'true';
  navToggle.setAttribute('aria-expanded', String(open));
  document.querySelector('.nav').classList.toggle('open', open);
});
// 完整静态表格可在无 JavaScript 时浏览；筛选与分页只作渐进增强。
const table = document.querySelector('#experiment-rows');
if (table) {
  const rows = [...table.querySelectorAll('tr')];
  const query = document.querySelector('#experiment-query');
  const category = document.querySelector('#experiment-category');
  const status = document.querySelector('#experiment-status');
  const counter = document.querySelector('#experiment-count');
  const previous = document.querySelector('#previous-page');
  const next = document.querySelector('#next-page');
  let page = 0;
  const size = 30;
  const params = new URLSearchParams(location.search);
  query.value = params.get('q') || '';
  category.value = params.get('category') || '';
  status.value = params.get('status') || '';
  const update = () => {
    const words = query.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const matches = rows.filter(row => (!category.value || row.dataset.category === category.value)
      && (!status.value || row.dataset.status === status.value)
      && words.every(word => row.textContent.toLocaleLowerCase().includes(word)));
    const pages = Math.max(1, Math.ceil(matches.length / size));
    page = Math.min(page, pages - 1);
    rows.forEach(row => { row.hidden = true; });
    matches.slice(page * size, (page + 1) * size).forEach(row => { row.hidden = false; });
    counter.textContent = `匹配 ${matches.length} / ${rows.length} 个实验 · 第 ${page + 1} / ${pages} 页`;
    document.querySelector('#empty-experiments').hidden = matches.length !== 0;
    previous.disabled = page === 0;
    next.disabled = page >= pages - 1;
    const state = new URLSearchParams();
    if (query.value) state.set('q', query.value);
    if (category.value) state.set('category', category.value);
    if (status.value) state.set('status', status.value);
    history.replaceState(null, '', location.pathname + (state.size ? `?${state}` : ''));
  };
  [query, category, status].forEach(input => input.addEventListener('input', () => { page = 0; update(); }));
  document.querySelector('#reset-filters').addEventListener('click', () => {
    query.value = category.value = status.value = ''; page = 0; update(); query.focus();
  });
  previous.addEventListener('click', () => { page--; update(); });
  next.addEventListener('click', () => { page++; update(); });
  document.querySelector('.pagination').hidden = false;
  update();
}
let searchIndex;
async function getIndex() {
  if (!searchIndex) {
    searchIndex = fetch(`${base}/assets/search-index.json`).then(response => {
      if (!response.ok) throw new Error('无法加载索引');
      return response.json();
    }).catch(error => { searchIndex = undefined; throw error; });
  }
  return searchIndex;
}
function bindSearch(input, results, counter) {
  let generation = 0;
  input.addEventListener('input', async () => {
    const current = ++generation;
    const words = input.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    results.replaceChildren();
    if (!words.length) { counter.textContent = '输入中文术语、参数名或实验 ID，例如：梯度累积、LoRA、optimizer.lr。'; return; }
    counter.textContent = '正在检索…';
    try {
      const index = await getIndex();
      if (current !== generation) return;
      const matches = index.map(item => {
        const text = `${item.title} ${item.text}`.toLocaleLowerCase();
        return { item, score: words.every(word => text.includes(word)) ? words.reduce((n, word) => n + (item.title.toLocaleLowerCase().includes(word) ? 8 : 1), 0) : 0 };
      }).filter(x => x.score).sort((a,b) => b.score - a.score);
      counter.textContent = matches.length ? `找到 ${matches.length} 条，展示相关度最高的 ${Math.min(50, matches.length)} 条。` : '没有找到匹配内容。试试更短的术语或实验 ID。';
      matches.slice(0,50).forEach(({item}) => {
        const li = document.createElement('li');
        const kind = document.createElement('small'); kind.textContent = item.kind;
        const heading = document.createElement('h3');
        const link = document.createElement('a'); link.href = item.url; link.textContent = item.title;
        const excerpt = document.createElement('p'); excerpt.textContent = item.excerpt;
        heading.append(link); li.append(kind,heading,excerpt); results.append(li);
      });
    } catch { if (current === generation) counter.textContent = '检索索引暂时不可用，请通过知识目录或实验目录浏览。'; }
  });
}
const dialog = document.querySelector('#search-dialog');
if (dialog) {
  const input = dialog.querySelector('input');
  bindSearch(input, dialog.querySelector('.search-results'), dialog.querySelector('[role="status"]'));
  document.querySelectorAll('[data-open-search]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault(); dialog.showModal(); input.focus();
  }));
  dialog.querySelector('[data-close-search]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && dialog.open) { event.preventDefault(); dialog.close(); }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); if (!dialog.open) dialog.showModal(); input.focus();
    }
  });
}
const pageSearch = document.querySelector('#page-search');
if (pageSearch) {
  bindSearch(pageSearch, document.querySelector('#page-results'), document.querySelector('#page-search-count'));
  pageSearch.value = new URLSearchParams(location.search).get('q') || '';
  if (pageSearch.value) pageSearch.dispatchEvent(new Event('input'));
}
document.querySelectorAll('.prose pre').forEach(pre => {
  const button = document.createElement('button'); button.type = 'button'; button.className = 'copy-code'; button.textContent = '复制代码';
  button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(pre.textContent); button.textContent = '已复制'; }
    catch { button.textContent = '请选中代码复制'; }
    setTimeout(() => { button.textContent = '复制代码'; }, 1800);
  });
  pre.after(button);
});
