from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

class AlayaliteConfig(BaseSettings):

    ALAYALITE_URL: Optional[str] = Field(
        default=None, description="The Access Key ID provided by Alayalite for API authentication."
    )