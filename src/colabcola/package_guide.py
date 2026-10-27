"""Static project inspection and reusable TOML answers for command-first packaging.

Never imports the workload, executes its help, resolves packages on the network,
or reads the packaging host's installed-package freeze.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tomllib

from .errors import SpecError

_FIELDS = {
    '': {'commands','project','include','exclude','name','output','inputs','remote_input_files',
         'environment','compute','outputs','limits','execution'},
    'environment': {'python','requirements','dependencies','runtime_requirements','wheels','isolated_python','env'},
    'compute': {'gpu','minimum_gpu_memory_gib','minimum_gpu_memory_bytes','high_mem'},
    'outputs': {'include','maximum_bytes','checkpoints','resume_arg','logs','completed'},
    'limits': {'timeout_minutes','timeout_seconds','max_seconds','max_runtimes','seed'},
    'execution': {'drive_parent','token_command','runtime_version','provider_backend'},
    'inputs': {'source','target'},
}
_IMPORT_NAMES = {'PIL':'pillow','cv2':'opencv-python','sklearn':'scikit-learn','yaml':'pyyaml',
                 'skimage':'scikit-image','bs4':'beautifulsoup4'}
_MISSING = object()


def _strings(value, label):
    if isinstance(value, str): return [value]
    if not isinstance(value, list) or any(not isinstance(x,str) or not x for x in value):
        raise SpecError(f'{label} must be a string or an array of strings')
    return value


def _validate(data):
    if not isinstance(data, dict): raise SpecError('answers must be a TOML table')
    for section, allowed in _FIELDS.items():
        value = data if not section else data.get(section, {})
        values = value if section == 'inputs' else [value]
        if section == 'inputs' and not isinstance(value, list):
            if value == {}: continue
            raise SpecError('inputs must be an array of source/target tables')
        for item in values:
            if not isinstance(item, dict): raise SpecError(f'{section or "answers"} must contain tables')
            extra = set(item) - allowed
            if extra: raise SpecError(f'unknown answer key: {section + "." if section else ""}{sorted(extra)[0]}')
    for key in ('include','exclude','remote_input_files'):
        if key in data: _strings(data[key],key)
    env=data.get('environment',{})
    for key in ('requirements','dependencies','runtime_requirements','wheels'):
        if key in env: _strings(env[key],'environment.'+key)
    for key in ('include','checkpoints','logs'):
        if key in data.get('outputs',{}): _strings(data['outputs'][key],'outputs.'+key)
    if 'completed' in data.get('outputs',{}):
        from .spec import _parse_managed_recovery
        _parse_managed_recovery({'checkpoint_pattern':'outputs/disabled.json','log_patterns':[], 'artifact_groups':data['outputs']['completed']})
    for key in ('commands',):
        if key in data:
            commands=data[key]
            if not isinstance(commands,list) or not commands: raise SpecError('commands must be nonempty argv arrays')
            for cmd in commands:
                if not isinstance(cmd,list) or not cmd or any(not isinstance(v,str) or '\0' in v for v in cmd):
                    raise SpecError('commands must be nonempty argv arrays of strings')
    for key in ('project','name','output'):
        if key in data and (not isinstance(data[key],str) or not data[key]): raise SpecError(key+' must be a nonempty string')
    for item in data.get('inputs',[]):
        for key in ('source','target'):
            if not isinstance(item.get(key),str) or not item[key]: raise SpecError('each input needs source and target strings')
    for section in ('compute','limits','outputs'):
        for key,value in data.get(section,{}).items():
            if key in ('minimum_gpu_memory_gib','minimum_gpu_memory_bytes','timeout_minutes','timeout_seconds','max_seconds','max_runtimes','seed','maximum_bytes'):
                if not isinstance(value,int) or isinstance(value,bool) or value < 0: raise SpecError(section+'.'+key+' must be a nonnegative integer')
    compute=data.get('compute',{})
    if 'high_mem' in compute and not isinstance(compute['high_mem'],bool): raise SpecError('compute.high_mem must be boolean')
    if 'gpu' in compute and not isinstance(compute['gpu'],(str,bool)): raise SpecError('compute.gpu must be boolean or accelerator name')
    if not isinstance(env.get('env',{}),dict) or any(not isinstance(v,str) for v in env.get('env',{}).values()):
        raise SpecError('environment.env must be a string-valued table')
    for section,keys in [('environment',('python','isolated_python')),('outputs',('resume_arg',)),('execution',('drive_parent','token_command','runtime_version','provider_backend'))]:
        for key in keys:
            if key in data.get(section,{}) and not isinstance(data[section][key],str): raise SpecError(section+'.'+key+' must be a string')


def _split_command(text):
    # Preserve backslashes in Windows paths while supporting quoted arguments.
    lexer=shlex.shlex(text,posix=True);lexer.whitespace_split=True;lexer.commenters='';lexer.escape=''
    result=list(lexer)
    if any(v in ('&&','||','|',';','>','<') for v in result):
        raise SpecError('shell operators are not supported; repeat --command for an ordered sequence')
    return result


def inspect_project(root, commands, includes, excludes):
    from .packager import _files
    ignores=root/'.colabcolaignore'
    if ignores.is_file(): excludes=list(excludes)+[v.strip() for v in ignores.read_text().splitlines() if v.strip() and not v.lstrip().startswith('#')]
    files=list(_files(root,excludes))
    import fnmatch
    if includes: files=[(p,r) for p,r in files if any(fnmatch.fnmatchcase(r,pat) for pat in includes)]
    imports=set(); gpu_hint=False
    local={Path(r).parts[0].split('.')[0] for _,r in files}
    local.update(Path(r).parts[1].split('.')[0] for _,r in files if r.startswith('src/') and len(Path(r).parts)>1)
    for path,relative in files:
        if path.suffix != '.py': continue
        if path.stat().st_size > 2*1024*1024: continue
        try: tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError,UnicodeError): continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.level == 0 and node.module: imports.add(node.module.split('.')[0])
            if isinstance(node,ast.Attribute) and node.attr == 'cuda': gpu_hint=True
    for cmd in commands:
        if len(cmd)>2 and cmd[1]=='-m': imports.add(cmd[2].split('.')[0])
        elif not re.fullmatch(r'python(?:3(?:\.\d+)?)?(?:\.exe)?',cmd[0]) and not cmd[0].endswith('.py'): imports.add({'torchrun':'torch'}.get(cmd[0],cmd[0]))
    external=sorted(imports - local - sys.stdlib_module_names - {'__future__'})
    dependencies=sorted({_IMPORT_NAMES.get(name,name) for name in external})
    pyproject={}
    if (root/'pyproject.toml').is_file():
        try: pyproject=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8-sig')).get('project',{})
        except tomllib.TOMLDecodeError as exc: raise SpecError(f'pyproject.toml: {exc}') from None
    requirements=[str(root/name) for name in ('requirements.lock','requirements.txt') if (root/name).is_file()]
    source_paths=[]
    for cmd in commands:
        if '-m' in cmd[:2]:
            i=cmd.index('-m')
            if len(cmd)>i+1:
                module=cmd[i+1].replace('.','/')
                if (root/'src'/(module+'.py')).is_file() or (root/'src'/module/'__main__.py').is_file(): source_paths=['src']
    return {'files':len(files),'imports':external,'dependencies':dependencies,'gpu_hint':gpu_hint,
            'requirements':requirements,'project_metadata':pyproject,'pythonpath':source_paths}


def _path(value, base):
    path=Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base/path))


def guide(args, arguments, cli_parser):
    if args.interactive and args.non_interactive: raise SpecError('choose either --interactive or --non-interactive')
    cwd=Path.cwd();base=cwd;data={}
    if args.answers:
        answer_path=_path(args.answers,cwd)
        if answer_path.suffix != '.colaml': raise SpecError('--colaml must name a .colaml TOML file')
        data=tomllib.loads(answer_path.read_text(encoding='utf-8-sig'));base=answer_path.parent
    _validate(data)
    prefix=arguments[:len(arguments)-len(args.command)] if args.command else arguments
    explicit=set()
    for action in cli_parser._actions:
        if any(token.split('=',1)[0] in action.option_strings for token in prefix): explicit.add(action.dest)
    def get(section,key,default=_MISSING):
        return (data if not section else data.get(section,{})).get(key,default)
    def assign(dest,section,key,convert=lambda v:v):
        value=get(section,key)
        if dest not in explicit and value is not _MISSING: setattr(args,dest,convert(value))
    questions=[]
    def ask(label,default=''):
        questions.append(label)
        if args.non_interactive: raise SpecError('missing answer: '+label+' (supply it in .colaml or enable questions)')
        try: value=input(f'{label} [{default}]: ').strip()
        except EOFError: raise SpecError('missing answer: '+label+'; stdin ended. Supply --colaml or use an interactive terminal') from None
        return value or str(default)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if args.command_strings and (command or args.commands): raise SpecError('use --command, --commands or trailing argv, not a combination')
    if command and args.commands: raise SpecError('use --commands or trailing argv, not both')
    if args.command_strings: commands=[_split_command(v) for v in args.command_strings]
    elif command: commands=[command]
    elif args.commands: commands=json.loads(args.commands.read_text(encoding='utf-8-sig'))
    else: commands=get('','commands',None)
    if not commands:
        commands=[_split_command(ask('Command to run in Colab'))]
    _validate({'commands':commands})
    commands=[list(cmd) for cmd in commands]
    root=_path(args.project,cwd) if args.project else _path(get('','project','.'),base)
    # An absolute standalone entry outside cwd supplies a useful project context.
    if not args.project and 'project' not in data and len(commands)==1:
        script=next((v for v in commands[0][:2] if v.endswith('.py')),None)
        if script and Path(script).is_absolute():
            try: Path(script).relative_to(root)
            except ValueError: root=Path(script).parent
    from .packager import _safe
    root=_safe(root)
    if not root.is_dir(): raise SpecError('project must be a directory')
    args.project=root
    assign('include','','include',lambda v:_strings(v,'include'))
    assign('exclude','','exclude',lambda v:_strings(v,'exclude'))
    info=inspect_project(root,commands,args.include,args.exclude)
    print(f'Project: {root}\nFound {info["files"]} project files; inspecting source without running it.',file=sys.stderr)
    if info['imports']: print('External imports: '+', '.join(info['imports']),file=sys.stderr)
    assign('name','','name')
    assign('output','','output',lambda v:_path(v,base))
    if args.output is None:
        first=commands[0]
        stem=first[2].split('.')[-1] if len(first)>2 and first[1]=='-m' else Path(first[1] if len(first)>1 and re.fullmatch(r'python(?:3(?:\.\d+)?)?(?:\.exe)?',first[0]) else first[0]).stem
        args.output=cwd/(re.sub(r'[^A-Za-z0-9_-]+','-',stem)+'.colajob')
    mappings=[]
    if 'data' in explicit: mappings=list(args.data)
    elif 'inputs' in data: mappings=[str(_path(item['source'],base))+'='+item['target'] for item in data['inputs']]
    else:
        suggested=[]
        for cmd in commands:
            for i,token in enumerate(cmd):
                if i and cmd[i-1] in ('--data','--dataset','--input','--input-dir','--config'):
                    source=_path(token,root)
                    if source.exists():
                        try: source.relative_to(root)
                        except ValueError:
                            target='data' if cmd[i-1] in ('--data','--dataset') else source.name
                            suggested.append(str(source)+'='+target)
        answer=ask('Additional inputs: SOURCE=TARGET separated by ; (none for no extra inputs)','; '.join(dict.fromkeys(suggested)) or 'none')
        mappings=[] if answer.lower()=='none' else [v.strip() for v in answer.split(';') if v.strip()]
    args.data=[]
    input_records=[]
    for mapping in mappings:
        if '=' not in mapping: raise SpecError('input needs SOURCE=TARGET')
        source,target=mapping.split('=',1);source=_path(source,cwd)
        args.data.append(str(source)+'='+target);input_records.append({'source':str(source),'target':target})
    # Rewrite only actual path arguments covered by an explicit input mapping.
    # Command executables and script paths are handled by the general packager.
    for cmd in commands:
        start=3 if len(cmd)>2 and cmd[1]=='-m' else 2 if re.fullmatch(r'python(?:3(?:\.\d+)?)?(?:\.exe)?',cmd[0]) else 1
        for i in range(start,len(cmd)):
            token=cmd[i]
            flag,separator,value=token.partition('=') if token.startswith('--') else ('','','')
            candidate=value if separator else token
            if candidate.startswith('-'): continue
            path=_path(candidate,root)
            if Path(candidate).is_absolute() and path.exists():
                try: relative=path.relative_to(root)
                except ValueError: pass
                else:
                    value=relative.as_posix()
                    cmd[i]=flag+'='+value if separator else value
            for item in sorted(input_records,key=lambda x:len(x['source']),reverse=True):
                try: relative=path.relative_to(item['source'])
                except ValueError: continue
                target=item['target']+('/'+relative.as_posix() if relative.parts else '')
                cmd[i]=flag+'='+target if separator else target
                break
    assign('requirements','environment','requirements',lambda v:[_path(x,base) for x in _strings(v,'environment.requirements')])
    assign('wheel','environment','wheels',lambda v:[_path(x,base) for x in _strings(v,'environment.wheels')])
    assign('runtime_requirement','environment','runtime_requirements',lambda v:_strings(v,'environment.runtime_requirements'))
    assign('remote_inputs','','remote_input_files',lambda v:[_path(x,base) for x in _strings(v,'remote_input_files')])
    declared=bool({'requirements','runtime_requirement'} & explicit) or any(k in data.get('environment',{}) for k in ('requirements','dependencies','runtime_requirements'))
    dependencies=_strings(get('environment','dependencies',[]),'environment.dependencies')
    if not declared:
        if info['requirements']:
            selected=info['requirements'][0]
            # Multiple declarations require a choice rather than merging incompatible locks.
            if len(info['requirements'])>1: selected=ask('Requirements file',selected)
            args.requirements=[_path(selected,root)]
        elif 'dependencies' in info['project_metadata']:
            dependencies=info['project_metadata']['dependencies']
        elif info['dependencies']:
            answer=ask('Python dependencies (comma separated; none to use runtime packages)',', '.join(info['dependencies']))
            dependencies=[] if answer.lower()=='none' else [v.strip() for v in answer.split(',') if v.strip()]
            if answer.lower()=='none': args.runtime_requirement=list(dict.fromkeys(args.runtime_requirement+info['dependencies']))
    dependencies=_strings(dependencies,'environment.dependencies')
    # Parse through the same requirement validator as files; no host installation.
    from packaging.requirements import Requirement,InvalidRequirement
    for dep in dependencies:
        try: req=Requirement(dep)
        except InvalidRequirement: raise SpecError('invalid dependency: '+dep) from None
        if req.url: raise SpecError('URL dependencies are not supported; use remote inputs or an embedded wheel')
    args.extra_requirements=dependencies
    assign('requires_python','environment','python')
    if 'requires_python' not in explicit and get('environment','python') is _MISSING:
        args.requires_python=info['project_metadata'].get('requires-python',args.requires_python)
    assign('isolated_python','environment','isolated_python')
    if args.isolated_python not in (None,'3.12'): raise SpecError('isolated Python currently supports 3.12 only')
    assign('env','environment','env',lambda v:[k+'='+value for k,value in v.items()])
    env=dict(v.split('=',1) for v in args.env)
    if info['pythonpath'] and 'PYTHONPATH' not in env: env['PYTHONPATH']=':'.join(info['pythonpath'])
    args.env=[key+'='+value for key,value in env.items()]
    args.allow_installed_modules=True
    if 'gpu' not in explicit:
        gpu=get('compute','gpu')
        if gpu is _MISSING: gpu=ask('Compute: NONE, T4, L4, A100, H100 or G4','T4' if info['gpu_hint'] else 'NONE')
        args.gpu='T4' if gpu is True else 'NONE' if gpu is False else str(gpu).upper()
    if args.gpu not in ('NONE','T4','L4','A100','H100','G4'): raise SpecError('unknown compute.gpu accelerator')
    assign('high_mem','compute','high_mem')
    assign('minimum_gpu_memory_bytes','compute','minimum_gpu_memory_bytes')
    if 'minimum_gpu_memory_bytes' not in explicit and get('compute','minimum_gpu_memory_bytes') is _MISSING and get('compute','minimum_gpu_memory_gib') is not _MISSING:
        args.minimum_gpu_memory_bytes=get('compute','minimum_gpu_memory_gib')*1024**3
    assign('artifact','outputs','include',lambda v:_strings(v,'outputs.include'))
    if 'artifact' not in explicit and get('outputs','include') is _MISSING:
        default='outputs/**'
        for cmd in commands:
            for i,token in enumerate(cmd[:-1]):
                if token in ('--output','--output-dir','--outdir') and not cmd[i+1].startswith('-'): default=cmd[i+1].rstrip('/')+'/**'
        answer=ask('Output globs to recover (comma separated)',default)
        args.artifact=[v.strip() for v in answer.split(',') if v.strip()]
    assign('artifact_max_bytes','outputs','maximum_bytes')
    assign('log','outputs','logs',lambda v:_strings(v,'outputs.logs'))
    assign('artifact_group','outputs','completed')
    checkpoints=get('outputs','checkpoints')
    if 'checkpoint' not in explicit:
        if checkpoints is _MISSING: checkpoints=ask('Checkpoint manifest glob (none if program cannot resume)','none')
        patterns=[] if checkpoints=='none' else _strings(checkpoints,'outputs.checkpoints')
        if len(patterns)>1: raise SpecError('only one checkpoint manifest glob is currently supported')
        args.checkpoint=patterns[0] if patterns else None
    assign('resume_arg','outputs','resume_arg')
    if args.checkpoint and not args.resume_arg and len(commands)==1:
        answer=ask('Option accepting a restored checkpoint (none if program reads COLABCOLA_RESUME_MANIFEST)','none')
        args.resume_arg=None if answer=='none' else answer
    assign('timeout','limits','timeout_seconds')
    if 'timeout' not in explicit and get('limits','timeout_seconds') is _MISSING:
        minutes=get('limits','timeout_minutes')
        if minutes is _MISSING: minutes=int(ask('Workload time limit in minutes','14'))
        args.timeout=minutes*60
    assign('max_seconds','limits','max_seconds')
    if 'max_seconds' not in explicit and get('limits','max_seconds') is _MISSING: args.max_seconds=max(900,args.timeout+60)
    assign('max_runtimes','limits','max_runtimes');assign('seed','limits','seed')
    for dest,key in [('drive_parent','drive_parent'),('token_command','token_command'),('runtime_version','runtime_version'),('provider_backend','provider_backend')]:
        assign(dest,'execution',key,lambda v:_path(v,base) if key=='token_command' else v)
    args.requirements=[_path(p,cwd) for p in args.requirements]
    args.wheel=[_path(p,cwd) for p in args.wheel]
    args.remote_inputs=[_path(p,cwd) for p in args.remote_inputs]
    args.output=_path(args.output,cwd)
    if args.token_command: args.token_command=_path(args.token_command,cwd)
    args.command_arrays=commands
    # Persist fully resolved answers. No host credential or executable inspection.
    resolved={'commands':commands,'project':str(root),'include':args.include,'exclude':args.exclude,
              'output':str(_path(args.output,cwd)),'inputs':input_records,'remote_input_files':[str(p) for p in args.remote_inputs],
              'environment':{'python':args.requires_python,'requirements':[str(p) for p in args.requirements],
                 'dependencies':dependencies,'runtime_requirements':args.runtime_requirement,'wheels':[str(p) for p in args.wheel], 'env':env},
              'compute':{'gpu':args.gpu,'high_mem':args.high_mem},
              'outputs':{'include':args.artifact,'maximum_bytes':args.artifact_max_bytes,
                 'checkpoints':[args.checkpoint] if args.checkpoint else [],'logs':args.log},
              'limits':{'timeout_seconds':args.timeout,'max_seconds':args.max_seconds,'max_runtimes':args.max_runtimes,'seed':args.seed},
              'execution':{'runtime_version':args.runtime_version,'provider_backend':args.provider_backend}}
    if args.artifact_group: resolved['outputs']['completed']=args.artifact_group
    if args.name: resolved['name']=args.name
    if args.isolated_python: resolved['environment']['isolated_python']=args.isolated_python
    if args.minimum_gpu_memory_bytes is not None: resolved['compute']['minimum_gpu_memory_bytes']=args.minimum_gpu_memory_bytes
    if args.resume_arg: resolved['outputs']['resume_arg']=args.resume_arg
    if args.drive_parent: resolved['execution'].update(drive_parent=args.drive_parent,token_command=str(args.token_command))
    print(f'Package {len(commands)} command(s) -> {args.output}; compute {args.gpu}; outputs {args.artifact}; workload {args.timeout}s. Packaging only.',file=sys.stderr)
    save_path=_path(args.save_answers,cwd) if args.save_answers else None
    if not args.non_interactive:
        if ask('Create this local package? y/n','y').lower() not in ('y','yes'): raise SpecError('packaging cancelled; no job created')
        if save_path is None:
            answer=ask('Save reusable answers (.colaml path, or none)','none')
            if answer.lower()!='none': save_path=_path(answer,cwd)
    if save_path:
        if save_path.suffix!='.colaml': raise SpecError('saved answers must end in .colaml')
        if save_path.exists(): raise SpecError('answers destination exists; choose a new --save-colaml')
        if not save_path.parent.is_dir(): raise SpecError('answers destination parent does not exist')
    return args,resolved,save_path,questions


def save_answers(path,answers):
    """Small TOML writer for our JSON-compatible answer types (no third-party dependency)."""
    def atom(value):
        if isinstance(value,bool): return 'true' if value else 'false'
        if isinstance(value,str): return json.dumps(value,ensure_ascii=False)
        if isinstance(value,int): return str(value)
        if isinstance(value,list): return '['+', '.join(atom(v) for v in value)+']'
        raise SpecError('unsupported answer value')
    lines=['# Colabcola answers: TOML; paths are relative to this file unless absolute.']
    def table(value,prefix=''):
        for key,item in value.items():
            if not isinstance(item,dict) and not (isinstance(item,list) and item and isinstance(item[0],dict)):
                lines.append(json.dumps(key)+' = '+atom(item))
        for key,item in value.items():
            name=prefix+'.'+json.dumps(key) if prefix else json.dumps(key)
            if isinstance(item,dict):
                lines.append('\n['+name+']');table(item,name)
            elif isinstance(item,list) and item and isinstance(item[0],dict):
                for record in item:
                    lines.append('\n[['+name+']]');table(record,name)
    table(answers)
    text='\n'.join(lines)+'\n'
    _validate(tomllib.loads(text))
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
