from pathlib import Path

p = Path("app/app.py"); t = p.read_text(encoding="utf-8")
if "_ui_kwargs" in t:
    print("already patched"); raise SystemExit

helpers = '''
THEME_KW = dict(primary_hue="green", secondary_hue="amber", neutral_hue="stone")
CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1100px !important; margin: auto !important;}
.card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 18px;
       background: var(--background-fill-secondary);}
#header h1 {margin-bottom: 0.1em;}
"""


def _ui_kwargs(fn):
    """Pass theme/css/footer options only where this Gradio version accepts them (they moved in v6)."""
    import inspect
    import gradio as gr
    ps = inspect.signature(fn).parameters
    kw = {}
    if "theme" in ps:
        kw["theme"] = gr.themes.Soft(**THEME_KW)
    if "css" in ps:
        kw["css"] = CSS
    if "footer_links" in ps:
        kw["footer_links"] = []
    return kw

'''
pairs = [
    ('\n\ndef resolve_climate(', helpers + '\ndef resolve_climate(', 1),
    ('with gr.Blocks(title="Crop Disease Advisory Assistant") as demo:\n        gr.Markdown(INTRO)',
     'with gr.Blocks(title="Crop Disease Advisory Assistant", **_ui_kwargs(gr.Blocks.__init__)) as demo:\n        gr.Markdown(INTRO, elem_id="header")', 1),
    ('answer = gr.Markdown(label="Answer")', 'answer = gr.Markdown(label="Answer", elem_classes="card")', 1),
    ('build_demo(pipe).queue().launch(share=args.share)',
     'import gradio as gr\n    build_demo(pipe).queue().launch(share=args.share, **_ui_kwargs(gr.Blocks.launch))', 1),
    ('"reflects disease risk. Answers are given in English.")',
     '"reflects disease risk. Answers are given in English.\\n\\n"\n         "<sub>University project prototype. Confirm advice with a local extension officer.</sub>")', 1),
]
for old, new, n in pairs:
    if old not in t:
        print("NOT FOUND:", old[:60].replace("\n", " "))
    t = t.replace(old, new, n)

start = t.find('        gr.Markdown("⚠️ **Prototype for a university project.**')
if start != -1:
    end = t.find("Always confirm with a local agricultural extension officer before treating crops.\")", start)
    end = t.find("\n", end) + 1
    t = t[:start] + t[end:]
else:
    print("NOT FOUND: footer disclaimer (maybe already removed or edited)")
p.write_text(t, encoding="utf-8"); print("patched app/app.py")
