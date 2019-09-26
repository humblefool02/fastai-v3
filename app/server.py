import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/open?id=1ThGp9A4kdkWknAorcttHvCEBm85Ne-Oq'
export_file_name = 'inception_transfer2.pkl'

classes = ['Banana Lady Finger', 'Melon Piel de Sapo', 'Strawberry',
       'Physalis with Husk', 'Apple Braeburn', 'Carambula',
       'Pineapple Mini', 'Onion Red Peeled', 'Potato Sweet', 'Beetroot',
       'Apple Red Yellow 2', 'Tomato 1', 'Pear Forelle', 'Chestnut',
       'Cocos', 'Apple Golden 3', 'Rambutan', 'Cherry Wax Red',
       'Grape Pink', 'Grape White 4', 'Tomato 2', 'Cantaloupe 1',
       'Onion Red', 'Cherry 1', 'Apple Granny Smith', 'Cactus fruit',
       'Raspberry', 'Kumquats', 'Passion Fruit', 'Grape White 2',
       'Pitahaya Red', 'Avocado', 'Peach Flat', 'Lemon', 'Blueberry',
       'Nectarine Flat', 'Apple Red Yellow 1', 'Walnut', 'Potato White',
       'Banana', 'Plum 2', 'Pepper Yellow', 'Pear Red', 'Mandarine',
       'Pear Williams', 'Cherry 2', 'Nut Pecan', 'Guava', 'Salak',
       'Physalis', 'Hazelnut', 'Lemon Meyer', 'Apple Golden 2', 'Plum 3',
       'Kohlrabi', 'Grapefruit Pink', 'Quince', 'Papaya', 'Huckleberry',
       'Peach', 'Cantaloupe 2', 'Pear Monster', 'Pear Kaiser', 'Pear',
       'Banana Red', 'Plum', 'Tomato Cherry Red', 'Pepper Green',
       'Nectarine', 'Clementine', 'Peach 2', 'Potato Red', 'Kaki',
       'Limes', 'Strawberry Wedge', 'Pomegranate', 'Nut Forest',
       'Apple Red 1', 'Grape White', 'Pineapple', 'Pepino',
       'Cherry Rainier', 'Pomelo Sweetie', 'Dates', 'Tomato 3','Mango Red', 'Onion White', 'Granadilla', 'Cauliflower',
       'Redcurrant', 'Pear Abate', 'Grape White 3', 'Apple Crimson Snow',
       'Orange', 'Mulberry', 'Apple Golden 1', 'Tomato Maroon',
       'Apple Red 3', 'Kiwi', 'Tangelo', 'Avocado ripe',
       'Apple Red Delicious', 'Eggplant', 'Apple Pink Lady', 'Mango',
       'Cherry Wax Yellow', 'Grapefruit White', 'Grape Blue', 'Mangostan',
       'Tomato Yellow', 'Tamarillo', 'Cherry Wax Black', 'Lychee',
       'Apricot', 'Pepper Red', 'Tomato 4', 'Apple Red 2',
       'Potato Red Washed', 'Maracuja', 'Ginger Root']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
