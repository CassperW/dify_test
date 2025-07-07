from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

class AlayaliteConfig(BaseSettings):

    URL: Optional[str] = Field(
        default=None, description="The Access Key ID provided by Alibaba Cloud for API authentication."
    )