import base64
from handler import handler

with open('assets/person.png', 'rb') as f:
    person_b64 = base64.b64encode(f.read()).decode()
with open('assets/sweater.png', 'rb') as f:
    cloth_b64 = base64.b64encode(f.read()).decode()

job = {'input': {'person': person_b64, 'cloth': cloth_b64}}

out = handler(job)
if 'result' in out:
    b64 = out['result']
    data = base64.b64decode(b64)
    with open('output.png', 'wb') as f:
        f.write(data)
    print('Saved output.png')
else:
    print('Error:', out)