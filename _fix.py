import pathlib, sys 
p = pathlib.Path('src/lightweight.py') 
lines = p.read_text('utf-8').splitlines() 
out = [] 
added = False 
