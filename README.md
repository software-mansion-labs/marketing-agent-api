# Marketing Agent Api :mag_right:

This package is a LangGraph multi-agent system searching the web through APIs, like Reddit API, for marketing opportunities for a product/company. It is a twin project of [Marketing Agent Web](https://github.com/software-mansion-labs/marketing-agent-web), which uses a search engine instead of specified APIs.

## How to use it

First, you have to describe the thing you want to advertise in `src/config.py`, in `DESCRIPTION_PROMPT` constant. As an example, we provide a description of our product—[React Native Executorch](https://docs.swmansion.com/react-native-executorch/). You can also control the number of iterations agents run before aggregating the results, or prompts of specific sub-agents. You also specify the timescope of posts searched and tags that might come in handy.

Once that's done, you create a `Crawler` object, as shown in `src/main.py`, and pass specific scrapers to it, which have to inherit from the `BaseScraper` class. The crawler is all set and you can run the search. It returns a list of websites suitable for advertisement, along with justifications of its picks.

### Scrapers

`Crawler` takes in a scraper tool as one of its arguments. This is the tool that the agents use to search through the website and load posts. It's customizable, and we provide an example tool in `src/tools`, a scraper for Reddit. If you decide to use a scraper that requires key(s), specify it in `.env`. 

#### How to get Reddit ID and secret?

Log in to your account.

Go to https://www.reddit.com/prefs/apps > apps > create an app

Select 'script', enter app name and redirect url (http://localhost:8080)

Click 'create app' - your client ID is right below your app name, and secret ID is in the 'secret' section.

### Models

By default, OpenAI models are available through `langchain-openai` dependency. Other models are also supported, but you need to install their packages to use them (see the [integrations page](https://docs.langchain.com/oss/python/integrations/providers/overview)). Once you've installed a specific package, you just change the name of the model in `src/config.py` accordingly and provide a key in `.env`.

