### FlapPytorchBird
`python dqn.py`
### Visdom For Windows quick fix
The problem is that windows doesn't allow "?" in filenames. A quick fix for windows users would be to delete the corresponding entry in ext_files in server.py

```python
%sfonts/glyphicons-halflings-regular.eot?#iefix' % bb:
'glyphicons-halflings-regular.eot?#iefix',

```
(lines 568-569)  
