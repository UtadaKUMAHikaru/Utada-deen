import os
import discord
import contextlib
from dotenv import load_dotenv
import pandas as pd
import os
import random
import re
import asyncio
from icecream import ic

import generator
import responseClient

load_dotenv()
TOKEN = os.getenv('DISCORD-BOT-UK-TOKEN')

utada_deen = generator.UtadaDeen(
    characters=[],
    response_length=40,
    run_name="full_text_small_run1"
    # run_name="small_uafcsc_better"
)

client = discord.Client()

sessions = {}

async def execute_commands(session, message):
    commands = {
        "save": session.write_history,
        "breakdown": session.breakdown_character,
        "users": session.show_characters,
        "add": session.add_character,
        "remove": session.remove_character,
        "label": session.label_emotion,
        "models": session.show_models,
        "useModel": session.switch_model,
        "shutup": session.stop_talking,
        "temperature": session.modify_temperature,
        "trumpOrAI": session.trump_or_ai,
        "answer": session.answer,
        "help": session.send_help
    }
    for command_name, command in commands.items():
        if message.content.startswith(f"!{command_name}"):
            await command(message)
            break

@client.event
async def on_message(message):
    ic(sessions)
    ic(message.content)
    ic(message.guild)
    ic(message.guild.name)
    ic(client.user)

    # 如果是bot发的
    if message.author == client.user and not message.content.startswith("!"):
        return

    if message.author == client.user and "Commands:" not in message.content:
        if "!free" in message.content:
            sessions[message.guild.name].start_talking()
        elif message.content.startswith("!knight"):
            await sessions[message.guild.name].knight_user(message, client)

    # 在sessions中添加
    if message.guild.name not in sessions:
        sessions[message.guild.name] = responseClient.ResponseClient(utada_deen)
        sessions[message.guild.name].load_characters(message)
        ic(sessions)

    if message.content.startswith("!"):
        await execute_commands(sessions[message.guild.name], message)
    else:
        await sessions[message.guild.name].respond_on_character(message)

    await message.channel.send(utada_deen.start_conversation(conversation=[], character="harry", filtered=True))

@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hi {member.name}, welcome to my Discord server!\n嗨{member.name}，欢迎来到我的Discord服务器!'
    )

@client.event
async def on_ready():
    print(f"Connected {client.user.name} - {client.user.id}\n------")

client.run(TOKEN)

