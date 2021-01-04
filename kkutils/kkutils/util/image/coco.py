import datetime


__all__ = [
    "coco_info",
]


def coco_info(
    description: str = "my coco dataset.",
    url: str = "http://test",
    version: str = "1.0",
    year: str = datetime.datetime.now().strftime("%Y"), 
    contributor: str = "Test",
    date_created: str = datetime.datetime.now().strftime("%Y/%m/%d")
):
    info = {}
    info["description"] = description
    info["url"] = url
    info["version"] = version
    info["year"] = year
    info["contributor"] = contributor
    info["date_created"] = date_created
    return info
