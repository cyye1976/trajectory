from pydantic import BaseModel


class anaAndVisDataRequestItem(BaseModel):
    model_filename: str
    group_id: int
    vis_options: list