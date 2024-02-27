import uvicorn
from fastapi import FastAPI
import sys

from pydantic import BaseModel

from api.entity.request import anaAndVisDataRequestItem

sys.path.append('..')
from api.loader.module import ApiModule

app = FastAPI()


@app.post("/analyseAndVisualizeData/")
def analyseAndVisualizeData(params: anaAndVisDataRequestItem):
    api_module = ApiModule(model_filename=params.model_filename, group_id=params.group_id)
    resultJsonPaths = api_module.inference()
    visPics = api_module.visualize(options=params.vis_options)
    anaData = api_module.load_result()
    result = {
        'resultJsonPaths': resultJsonPaths,
        'visPics': visPics,
        'anaData': anaData
    }
    return result



class Item(BaseModel):
    params: str

@app.post("/sayHello/")
def sayHello(item: Item):
    return {'message': item.params}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
