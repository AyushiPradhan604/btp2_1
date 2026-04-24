import os
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

def render_poster(data_contract: dict, output_image: str = "poster.png", template_name: str = "poster.html", critic_css: str = ""):
    """Generates poster image from HTML using Playwright."""
    
    # Render HTML
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"Failed to load template {template_name}: {e}")
        return False
        
    html_content = template.render(data=data_contract, critic_css=critic_css)
    
    html_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'temp_render.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    # Capture with Playwright
    try:
        with sync_playwright() as p:
            # Using chromium
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 3840, "height": 2880},
                device_scale_factor=1 # Exact size
            )
            
            # Use absolute path to ensure local assets load
            abs_html_path = f"file://{os.path.abspath(html_path)}"
            page.goto(abs_html_path, wait_until="networkidle")
            
            page.screenshot(path=output_image)
            browser.close()
    except Exception as e:
        print(f"Playwright render failed: {e}. Ensure 'playwright install chromium' has been run.")
        return False
        
    # Cleanup temp html
    if os.path.exists(html_path):
        os.remove(html_path)
        
    return True
