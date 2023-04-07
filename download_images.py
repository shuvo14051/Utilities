"""
pip install simple_image_download==0.4
to make this work must download this version
latest version does not work
"""

from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords = ["strawberry"]

for kw in keywords:
	response().download(kw,50)
