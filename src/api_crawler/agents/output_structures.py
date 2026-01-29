from pydantic import BaseModel, Field


class PostHeader(BaseModel):
    """Post title and URL."""

    title: str = Field(description="post title")
    link: str = Field(description="URL of the post")


class Post(BaseModel):
    """Post header and content."""

    header: PostHeader = Field(description="post identification")
    content: str = Field(description="post contents")


class PostChoice(BaseModel):
    """Post and justification why it's a good pick."""

    post: PostHeader = Field(description="post")
    justification: str = Field(description="why it's a good place to advertise")


class PostChoiceList(BaseModel):
    """List of Post choices."""

    posts: list[PostChoice] = Field(description="list of posts")


class Critique(BaseModel):
    """Content suitability critique."""

    ad_upsides: str = Field(description="upsides of advertising suitability")
    ad_downsides: str = Field(description="downsides of advertising suitability")


class PostCritique(BaseModel):
    """Post suitability critique."""

    post: PostHeader = Field(description="post")
    critique: Critique = Field(description="critique of its suitability")


class PostsToLoad(BaseModel):
    """Posts to load."""

    posts: list[PostHeader] = Field(description="list of posts to load")
