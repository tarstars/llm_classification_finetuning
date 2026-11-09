"""Conservative local numerical claims; no eval, symbolic assumptions or verdicts."""
import ast
from collections import Counter
from fractions import Fraction
import re

RISK=re.compile(r'\b(?:wrong|incorrect|false|contradiction|counterexample|hypothetical|suppose|assume|pretend|fictional|orwell|error)\b|not\s+(?:true|equal|correct)',re.I)
MATH=re.compile(r'\b(?:arithmetic|calculate|sum|product|multiply|divide|equation|math|solve|evaluate|simplify|compute|plus|minus|add|subtract|factor|proof)\b|\d\s*[+*/^×÷−-]\s*\d',re.I)
PREFIX=re.compile(r'^(?:(?:so|thus|therefore|hence|we get|we have|this gives|the result is|the answer is|result|answer)\s*[:,]?\s*)',re.I)
ALLOWED=re.compile(r'^[\d\s.()+*/^×÷−-]+$')


def rational(expression):
    text=expression.strip().replace('×','*').replace('÷','/').replace('−','-').replace('^','**')
    if not text or len(text)>160 or not ALLOWED.fullmatch(text):raise ValueError('unsupported expression')
    tree=ast.parse(text,mode='eval')
    if len(list(ast.walk(tree)))>48:raise ValueError('expression size')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in [int,float]:
            token=ast.get_source_segment(text,n)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',token):raise ValueError('literal')
            value=Fraction(token)
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            value=visit(n.operand);value=-value if isinstance(n.op,ast.USub) else value
        elif isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):value=a+b
            elif isinstance(n.op,ast.Sub):value=a-b
            elif isinstance(n.op,ast.Mult):value=a*b
            elif isinstance(n.op,ast.Div):value=a/b
            elif isinstance(n.op,ast.Pow) and b.denominator==1 and abs(b)<=8:value=a**int(b)
            else:raise ValueError('operator')
        else:raise ValueError('AST node')
        if abs(value)>10**18 or value.denominator.bit_length()>160:raise ValueError('magnitude')
        return value
    return visit(tree.body)


def answer_claims(prompt,text):
    if not isinstance(text,str) or len(text)>300000:return [],'response_limit'
    if not MATH.search(prompt):return [],'nonmath_request'
    if RISK.search(prompt) or RISK.search(text):return [],'ambiguous_context'
    clean=re.sub(r'```.*?```','',text,flags=re.S)
    clean=re.sub(r'`[^`]*`','',clean)
    candidates=[]
    for m in re.finditer(r'\$\$?([^$]{1,256})\$\$?|\\\(([^\\]{1,256})\\\)',clean):
        candidates.append(m.group(1) or m.group(2))
    for line in clean.splitlines():
        line=line.strip()
        if line.startswith(('- ','>','"',"'")):continue
        line=line.strip('* $')
        line=PREFIX.sub('',line)
        if line.endswith('.'):line=line[:-1]
        candidates.append(line)
    output=[];seen=set()
    for claim in candidates:
        claim=claim.strip()
        if claim in seen or claim.count('=')!=1 or len(claim)>256:continue
        seen.add(claim);left,right=[x.strip() for x in claim.split('=')]
        try:a,b=rational(left),rational(right)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):continue
        tolerance=Fraction(0)
        decimal=re.fullmatch(r'[+-]?\d*\.(\d+)',right)
        if decimal:tolerance=Fraction(1,2*10**len(decimal.group(1)))
        output.append(dict(claim=claim,left=left,right=right,error=abs(a-b)>tolerance,
            left_value=str(a),right_value=str(b),rounding_tolerance=str(tolerance)))
        if len(output)>=32:break
    return output,'checked' if output else 'unsupported_or_absent'


def row_feature(prompts,answers_a,answers_b):
    if not(len(prompts)==len(answers_a)==len(answers_b)) or not prompts:return dict(feature=0,reason='unaligned',claims_a=[],claims_b=[])
    ca,cb,reasons=[],[],Counter()
    for p,a,b in zip(prompts,answers_a,answers_b):
        p='' if p is None else str(p)
        for answer,target in [(a,ca),(b,cb)]:
            claims,reason=answer_claims(p,'' if answer is None else str(answer));target.extend(claims);reasons[reason]+=1
    ea,eb=sum(x['error'] for x in ca),sum(x['error'] for x in cb)
    return dict(feature=eb-ea,claims_a=ca,claims_b=cb,error_count_a=ea,error_count_b=eb,reasons=dict(reasons))
