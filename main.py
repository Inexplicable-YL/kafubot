from dotenv import load_dotenv

from kafubot import Bot

load_dotenv()


bot = Bot(config_file="config.toml")


if __name__ == "__main__":
    bot.run()
