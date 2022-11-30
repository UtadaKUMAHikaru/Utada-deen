# bot.py

import os
import random
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD-BOT-UK-TOKEN')

# intents = discord.Intents.default()
# intents.members = True # Subscribe to the privileged members intent.
# intents.guild_messages = True 
# intents.guild_reactions = True 
# intents.guilds = True 
# intents.typing = True 
# bot = commands.Bot(command_prefix='!', intents=intents)
bot = commands.Bot(command_prefix='!')

@bot.command(name='99')
async def nine_nine(ctx):
    brooklyn_99_quotes = [
        'I\'m the human form of the 💯 emoji.',
        'Bingpot!',
        (
            'Cool. Cool cool cool cool cool cool cool, '
            'no doubt no doubt no doubt no doubt.'
        ),
    ]

    response = random.choice(brooklyn_99_quotes)
    await ctx.send(response)

bot.run(TOKEN)