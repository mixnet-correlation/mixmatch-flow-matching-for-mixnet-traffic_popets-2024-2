­Ј
┬Ц
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8╔­
ю
"deep_coffea_model/FeaturesVec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"deep_coffea_model/FeaturesVec/bias
Ћ
6deep_coffea_model/FeaturesVec/bias/Read/ReadVariableOpReadVariableOp"deep_coffea_model/FeaturesVec/bias*
_output_shapes
:@*
dtype0
Ц
$deep_coffea_model/FeaturesVec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*5
shared_name&$deep_coffea_model/FeaturesVec/kernel
ъ
8deep_coffea_model/FeaturesVec/kernel/Read/ReadVariableOpReadVariableOp$deep_coffea_model/FeaturesVec/kernel*
_output_shapes
:	ђ@*
dtype0
Ъ
#deep_coffea_model/block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*4
shared_name%#deep_coffea_model/block3_conv2/bias
ў
7deep_coffea_model/block3_conv2/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block3_conv2/bias*
_output_shapes	
:ђ*
dtype0
г
%deep_coffea_model/block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*6
shared_name'%deep_coffea_model/block3_conv2/kernel
Ц
9deep_coffea_model/block3_conv2/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block3_conv2/kernel*$
_output_shapes
:ђђ*
dtype0
Ъ
#deep_coffea_model/block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*4
shared_name%#deep_coffea_model/block3_conv1/bias
ў
7deep_coffea_model/block3_conv1/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block3_conv1/bias*
_output_shapes	
:ђ*
dtype0
г
%deep_coffea_model/block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*6
shared_name'%deep_coffea_model/block3_conv1/kernel
Ц
9deep_coffea_model/block3_conv1/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block3_conv1/kernel*$
_output_shapes
:ђђ*
dtype0
Ъ
#deep_coffea_model/block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*4
shared_name%#deep_coffea_model/block2_conv2/bias
ў
7deep_coffea_model/block2_conv2/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block2_conv2/bias*
_output_shapes	
:ђ*
dtype0
г
%deep_coffea_model/block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*6
shared_name'%deep_coffea_model/block2_conv2/kernel
Ц
9deep_coffea_model/block2_conv2/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block2_conv2/kernel*$
_output_shapes
:ђђ*
dtype0
Ъ
#deep_coffea_model/block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*4
shared_name%#deep_coffea_model/block2_conv1/bias
ў
7deep_coffea_model/block2_conv1/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block2_conv1/bias*
_output_shapes	
:ђ*
dtype0
Ф
%deep_coffea_model/block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*6
shared_name'%deep_coffea_model/block2_conv1/kernel
ц
9deep_coffea_model/block2_conv1/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block2_conv1/kernel*#
_output_shapes
:@ђ*
dtype0
ъ
#deep_coffea_model/block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#deep_coffea_model/block1_conv2/bias
Ќ
7deep_coffea_model/block1_conv2/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block1_conv2/bias*
_output_shapes
:@*
dtype0
ф
%deep_coffea_model/block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%deep_coffea_model/block1_conv2/kernel
Б
9deep_coffea_model/block1_conv2/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block1_conv2/kernel*"
_output_shapes
:@@*
dtype0
ъ
#deep_coffea_model/block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#deep_coffea_model/block1_conv1/bias
Ќ
7deep_coffea_model/block1_conv1/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block1_conv1/bias*
_output_shapes
:@*
dtype0
ф
%deep_coffea_model/block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%deep_coffea_model/block1_conv1/kernel
Б
9deep_coffea_model/block1_conv1/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block1_conv1/kernel*"
_output_shapes
: @*
dtype0
ъ
#deep_coffea_model/block0_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#deep_coffea_model/block0_conv2/bias
Ќ
7deep_coffea_model/block0_conv2/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block0_conv2/bias*
_output_shapes
: *
dtype0
ф
%deep_coffea_model/block0_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%deep_coffea_model/block0_conv2/kernel
Б
9deep_coffea_model/block0_conv2/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block0_conv2/kernel*"
_output_shapes
:  *
dtype0
ъ
#deep_coffea_model/block0_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#deep_coffea_model/block0_conv1/bias
Ќ
7deep_coffea_model/block0_conv1/bias/Read/ReadVariableOpReadVariableOp#deep_coffea_model/block0_conv1/bias*
_output_shapes
: *
dtype0
ф
%deep_coffea_model/block0_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%deep_coffea_model/block0_conv1/kernel
Б
9deep_coffea_model/block0_conv1/kernel/Read/ReadVariableOpReadVariableOp%deep_coffea_model/block0_conv1/kernel*"
_output_shapes
: *
dtype0
ѓ
serving_default_input_1Placeholder*+
_output_shapes
:         d*
dtype0* 
shape:         d
Ѕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1%deep_coffea_model/block0_conv1/kernel#deep_coffea_model/block0_conv1/bias%deep_coffea_model/block0_conv2/kernel#deep_coffea_model/block0_conv2/bias%deep_coffea_model/block1_conv1/kernel#deep_coffea_model/block1_conv1/bias%deep_coffea_model/block1_conv2/kernel#deep_coffea_model/block1_conv2/bias%deep_coffea_model/block2_conv1/kernel#deep_coffea_model/block2_conv1/bias%deep_coffea_model/block2_conv2/kernel#deep_coffea_model/block2_conv2/bias%deep_coffea_model/block3_conv1/kernel#deep_coffea_model/block3_conv1/bias%deep_coffea_model/block3_conv2/kernel#deep_coffea_model/block3_conv2/bias$deep_coffea_model/FeaturesVec/kernel"deep_coffea_model/FeaturesVec/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *1
f,R*
(__inference_signature_wrapper_6539620051

NoOpNoOp
╠d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Єd
value§cBЩc Bзc
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
conv_layers
	pool_layers

dropout_layers
flatten
	dense

signatures*
і
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17*
і
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17*
* 
░
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
%trace_0
&trace_1
'trace_2
(trace_3* 
6
)trace_0
*trace_1
+trace_2
,trace_3* 
* 
<
-0
.1
/2
03
14
25
36
47*

50
61
72
83* 

90
:1
;2* 
ј
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
д
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

kernel
bias*

Hserving_default* 
e_
VARIABLE_VALUE%deep_coffea_model/block0_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_coffea_model/block0_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%deep_coffea_model/block0_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_coffea_model/block0_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%deep_coffea_model/block1_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_coffea_model/block1_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%deep_coffea_model/block1_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_coffea_model/block1_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%deep_coffea_model/block2_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_coffea_model/block2_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%deep_coffea_model/block2_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#deep_coffea_model/block2_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%deep_coffea_model/block3_conv1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#deep_coffea_model/block3_conv1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%deep_coffea_model/block3_conv2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#deep_coffea_model/block3_conv2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$deep_coffea_model/FeaturesVec/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"deep_coffea_model/FeaturesVec/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
ѓ
90
:1
;2
-3
.4
/5
06
17
28
39
410
511
612
713
814
15
16*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
╚
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
bias
 O_jit_compiled_convolution_op*
╚
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias
 V_jit_compiled_convolution_op*
╚
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias
 ]_jit_compiled_convolution_op*
╚
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op*
╚
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kernel
bias
 k_jit_compiled_convolution_op*
╚
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

kernel
bias
 r_jit_compiled_convolution_op*
╚
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

kernel
bias
 y_jit_compiled_convolution_op*
╔
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias
!ђ_jit_compiled_convolution_op*
ћ
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
ё	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses* 
ћ
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses* 
ћ
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses* 
ћ
Њ	variables
ћtrainable_variables
Ћregularization_losses
ќ	keras_api
Ќ__call__
+ў&call_and_return_all_conditional_losses* 
г
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ъ_random_generator* 
г
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses
д_random_generator* 
г
Д	variables
еtrainable_variables
Еregularization_losses
ф	keras_api
Ф__call__
+г&call_and_return_all_conditional_losses
Г_random_generator* 
* 
* 
* 
ќ
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

│trace_0* 

┤trace_0* 

0
1*

0
1*
* 
ў
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

║trace_0* 

╗trace_0* 
* 

0
1*

0
1*
* 
ў
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

┴trace_0* 

┬trace_0* 
* 

0
1*

0
1*
* 
ў
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

╚trace_0* 

╔trace_0* 
* 

0
1*

0
1*
* 
ў
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

¤trace_0* 

лtrace_0* 
* 

0
1*

0
1*
* 
ў
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

оtrace_0* 

Оtrace_0* 
* 

0
1*

0
1*
* 
ў
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

Пtrace_0* 

яtrace_0* 
* 

0
1*

0
1*
* 
ў
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

Сtrace_0* 

тtrace_0* 
* 

0
1*

0
1*
* 
ў
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

вtrace_0* 

Вtrace_0* 
* 

0
1*

0
1*
* 
ў
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ыtrace_0* 

зtrace_0* 
* 
* 
* 
* 
ю
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

щtrace_0* 

Щtrace_0* 
* 
* 
* 
ю
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses* 

ђtrace_0* 

Ђtrace_0* 
* 
* 
* 
ю
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses* 

Єtrace_0* 

ѕtrace_0* 
* 
* 
* 
ю
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Њ	variables
ћtrainable_variables
Ћregularization_losses
Ќ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses* 

јtrace_0* 

Јtrace_0* 
* 
* 
* 
ю
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses* 

Ћtrace_0
ќtrace_1* 

Ќtrace_0
ўtrace_1* 
* 
* 
* 
* 
ю
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

ъtrace_0
Ъtrace_1* 

аtrace_0
Аtrace_1* 
* 
* 
* 
* 
ю
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
Д	variables
еtrainable_variables
Еregularization_losses
Ф__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses* 

Дtrace_0
еtrace_1* 

Еtrace_0
фtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╚

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9deep_coffea_model/block0_conv1/kernel/Read/ReadVariableOp7deep_coffea_model/block0_conv1/bias/Read/ReadVariableOp9deep_coffea_model/block0_conv2/kernel/Read/ReadVariableOp7deep_coffea_model/block0_conv2/bias/Read/ReadVariableOp9deep_coffea_model/block1_conv1/kernel/Read/ReadVariableOp7deep_coffea_model/block1_conv1/bias/Read/ReadVariableOp9deep_coffea_model/block1_conv2/kernel/Read/ReadVariableOp7deep_coffea_model/block1_conv2/bias/Read/ReadVariableOp9deep_coffea_model/block2_conv1/kernel/Read/ReadVariableOp7deep_coffea_model/block2_conv1/bias/Read/ReadVariableOp9deep_coffea_model/block2_conv2/kernel/Read/ReadVariableOp7deep_coffea_model/block2_conv2/bias/Read/ReadVariableOp9deep_coffea_model/block3_conv1/kernel/Read/ReadVariableOp7deep_coffea_model/block3_conv1/bias/Read/ReadVariableOp9deep_coffea_model/block3_conv2/kernel/Read/ReadVariableOp7deep_coffea_model/block3_conv2/bias/Read/ReadVariableOp8deep_coffea_model/FeaturesVec/kernel/Read/ReadVariableOp6deep_coffea_model/FeaturesVec/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference__traced_save_6539621036
█
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%deep_coffea_model/block0_conv1/kernel#deep_coffea_model/block0_conv1/bias%deep_coffea_model/block0_conv2/kernel#deep_coffea_model/block0_conv2/bias%deep_coffea_model/block1_conv1/kernel#deep_coffea_model/block1_conv1/bias%deep_coffea_model/block1_conv2/kernel#deep_coffea_model/block1_conv2/bias%deep_coffea_model/block2_conv1/kernel#deep_coffea_model/block2_conv1/bias%deep_coffea_model/block2_conv2/kernel#deep_coffea_model/block2_conv2/bias%deep_coffea_model/block3_conv1/kernel#deep_coffea_model/block3_conv1/bias%deep_coffea_model/block3_conv2/kernel#deep_coffea_model/block3_conv2/bias$deep_coffea_model/FeaturesVec/kernel"deep_coffea_model/FeaturesVec/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ */
f*R(
&__inference__traced_restore_6539621100л┴
у
б
1__inference_block1_conv1_layer_call_fn_6539620660

inputs
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:          : @:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539611517s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
н
ь
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539611517
inputs_0A
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : *~
backward_function_namedb__inference___backward_block1_conv1_layer_call_and_return_conditional_losses_6539611481_653961151820
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
╬	
љ
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612383
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block3_pool_layer_call_and_return_conditional_losses_6539612362_6539612384:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
н
ь
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539611182
inputs_0A
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cњ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c: : *~
backward_function_namedb__inference___backward_block0_conv1_layer_call_and_return_conditional_losses_6539611146_653961118320
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c
 
_user_specified_nameinputs
И
J
.__inference_flatten_1_layer_call_fn_6539620559

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539612395a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ФM
Ъ
&__inference__traced_restore_6539621100
file_prefixL
6assignvariableop_deep_coffea_model_block0_conv1_kernel: D
6assignvariableop_1_deep_coffea_model_block0_conv1_bias: N
8assignvariableop_2_deep_coffea_model_block0_conv2_kernel:  D
6assignvariableop_3_deep_coffea_model_block0_conv2_bias: N
8assignvariableop_4_deep_coffea_model_block1_conv1_kernel: @D
6assignvariableop_5_deep_coffea_model_block1_conv1_bias:@N
8assignvariableop_6_deep_coffea_model_block1_conv2_kernel:@@D
6assignvariableop_7_deep_coffea_model_block1_conv2_bias:@O
8assignvariableop_8_deep_coffea_model_block2_conv1_kernel:@ђE
6assignvariableop_9_deep_coffea_model_block2_conv1_bias:	ђQ
9assignvariableop_10_deep_coffea_model_block2_conv2_kernel:ђђF
7assignvariableop_11_deep_coffea_model_block2_conv2_bias:	ђQ
9assignvariableop_12_deep_coffea_model_block3_conv1_kernel:ђђF
7assignvariableop_13_deep_coffea_model_block3_conv1_bias:	ђQ
9assignvariableop_14_deep_coffea_model_block3_conv2_kernel:ђђF
7assignvariableop_15_deep_coffea_model_block3_conv2_bias:	ђK
8assignvariableop_16_deep_coffea_model_featuresvec_kernel:	ђ@D
6assignvariableop_17_deep_coffea_model_featuresvec_bias:@
identity_19ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueЂB■B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHќ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOp6assignvariableop_deep_coffea_model_block0_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_1AssignVariableOp6assignvariableop_1_deep_coffea_model_block0_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_2AssignVariableOp8assignvariableop_2_deep_coffea_model_block0_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_3AssignVariableOp6assignvariableop_3_deep_coffea_model_block0_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_4AssignVariableOp8assignvariableop_4_deep_coffea_model_block1_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_5AssignVariableOp6assignvariableop_5_deep_coffea_model_block1_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_6AssignVariableOp8assignvariableop_6_deep_coffea_model_block1_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_7AssignVariableOp6assignvariableop_7_deep_coffea_model_block1_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_8AssignVariableOp8assignvariableop_8_deep_coffea_model_block2_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_9AssignVariableOp6assignvariableop_9_deep_coffea_model_block2_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_10AssignVariableOp9assignvariableop_10_deep_coffea_model_block2_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_11AssignVariableOp7assignvariableop_11_deep_coffea_model_block2_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_12AssignVariableOp9assignvariableop_12_deep_coffea_model_block3_conv1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_13AssignVariableOp7assignvariableop_13_deep_coffea_model_block3_conv1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_14AssignVariableOp9assignvariableop_14_deep_coffea_model_block3_conv2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_15AssignVariableOp7assignvariableop_15_deep_coffea_model_block3_conv2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_16AssignVariableOp8assignvariableop_16_deep_coffea_model_featuresvec_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_17AssignVariableOp6assignvariableop_17_deep_coffea_model_featuresvec_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 █
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: ╚
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤
g
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539620839

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
¤
g
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539611034

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ў
l
3__inference_block0_dropout_layer_call_fn_6539620888

inputs
identityѕбStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539616099s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
ш
l
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539612059

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ђ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬	
љ
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539614616
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block0_pool_layer_call_and_return_conditional_losses_6539614594_6539614617:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
▀
ъ
L__inference_block3_conv1_layer_call_and_return_conditional_losses_6539620796

inputsC
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ѓ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
С
­
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612869
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block3_conv2_layer_call_and_return_conditional_losses_6539612830_653961287020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
l
3__inference_block1_dropout_layer_call_fn_6539620915

inputs
identityѕбStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539616046s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
С
­
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539611978
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block2_conv2_layer_call_and_return_conditional_losses_6539611942_653961197920
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
х

m
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539615993

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:б
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ђ*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ђt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ђn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ђ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ы
j
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539613947

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         @_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @*ђ
backward_function_namefd__inference___backward_block1_dropout_layer_call_and_return_conditional_losses_6539613942_6539613948:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
н
ь
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539614747
inputs_0A
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c : : *~
backward_function_namedb__inference___backward_block0_conv2_layer_call_and_return_conditional_losses_6539614708_653961474820
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c 
 
_user_specified_nameinputs
ю
l
3__inference_block2_dropout_layer_call_fn_6539620942

inputs
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539615993t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
нГ
П
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620397	
inputN
8block0_conv1_conv1d_expanddims_1_readvariableop_resource: :
,block0_conv1_biasadd_readvariableop_resource: N
8block0_conv2_conv1d_expanddims_1_readvariableop_resource:  :
,block0_conv2_biasadd_readvariableop_resource: N
8block1_conv1_conv1d_expanddims_1_readvariableop_resource: @:
,block1_conv1_biasadd_readvariableop_resource:@N
8block1_conv2_conv1d_expanddims_1_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@O
8block2_conv1_conv1d_expanddims_1_readvariableop_resource:@ђ;
,block2_conv1_biasadd_readvariableop_resource:	ђP
8block2_conv2_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block2_conv2_biasadd_readvariableop_resource:	ђP
8block3_conv1_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block3_conv1_biasadd_readvariableop_resource:	ђP
8block3_conv2_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block3_conv2_biasadd_readvariableop_resource:	ђ=
*featuresvec_matmul_readvariableop_resource:	ђ@9
+featuresvec_biasadd_readvariableop_resource:@
identityѕб"FeaturesVec/BiasAdd/ReadVariableOpб!FeaturesVec/MatMul/ReadVariableOpб#block0_conv1/BiasAdd/ReadVariableOpб/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block0_conv2/BiasAdd/ReadVariableOpб/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block1_conv1/BiasAdd/ReadVariableOpб/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block1_conv2/BiasAdd/ReadVariableOpб/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block2_conv1/BiasAdd/ReadVariableOpб/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block2_conv2/BiasAdd/ReadVariableOpб/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block3_conv1/BiasAdd/ReadVariableOpб/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block3_conv2/BiasAdd/ReadVariableOpб/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         у
strided_sliceStridedSliceinputstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         №
strided_slice_1StridedSliceinputstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskr
subSubstrided_slice:output:0strided_slice_1:output:0*
T0*+
_output_shapes
:         cm
"block0_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ю
block0_conv1/Conv1D/ExpandDims
ExpandDimssub:z:0+block0_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cг
/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block0_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0f
$block0_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block0_conv1/Conv1D/ExpandDims_1
ExpandDims7block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block0_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: М
block0_conv1/Conv1DConv2D'block0_conv1/Conv1D/ExpandDims:output:0)block0_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
џ
block0_conv1/Conv1D/SqueezeSqueezeblock0_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ї
#block0_conv1/BiasAdd/ReadVariableOpReadVariableOp,block0_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
block0_conv1/BiasAddBiasAdd$block0_conv1/Conv1D/Squeeze:output:0+block0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c n
block0_conv1/ReluRelublock0_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         c m
"block0_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ┤
block0_conv2/Conv1D/ExpandDims
ExpandDimsblock0_conv1/Relu:activations:0+block0_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c г
/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block0_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0f
$block0_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block0_conv2/Conv1D/ExpandDims_1
ExpandDims7block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block0_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  М
block0_conv2/Conv1DConv2D'block0_conv2/Conv1D/ExpandDims:output:0)block0_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
џ
block0_conv2/Conv1D/SqueezeSqueezeblock0_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ї
#block0_conv2/BiasAdd/ReadVariableOpReadVariableOp,block0_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
block0_conv2/BiasAddBiasAdd$block0_conv2/Conv1D/Squeeze:output:0+block0_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c n
block0_conv2/ReluRelublock0_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         c \
block0_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ц
block0_pool/ExpandDims
ExpandDimsblock0_conv2/Relu:activations:0#block0_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c Ф
block0_pool/MaxPoolMaxPoolblock0_pool/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
Ѕ
block0_pool/SqueezeSqueezeblock0_pool/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
w
block0_dropout/IdentityIdentityblock0_pool/Squeeze:output:0*
T0*+
_output_shapes
:          m
"block1_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block1_conv1/Conv1D/ExpandDims
ExpandDims block0_dropout/Identity:output:0+block1_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          г
/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0f
$block1_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block1_conv1/Conv1D/ExpandDims_1
ExpandDims7block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block1_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @М
block1_conv1/Conv1DConv2D'block1_conv1/Conv1D/ExpandDims:output:0)block1_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
џ
block1_conv1/Conv1D/SqueezeSqueezeblock1_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0е
block1_conv1/BiasAddBiasAdd$block1_conv1/Conv1D/Squeeze:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @n
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         @m
"block1_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ┤
block1_conv2/Conv1D/ExpandDims
ExpandDimsblock1_conv1/Relu:activations:0+block1_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @г
/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0f
$block1_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block1_conv2/Conv1D/ExpandDims_1
ExpandDims7block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block1_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@М
block1_conv2/Conv1DConv2D'block1_conv2/Conv1D/ExpandDims:output:0)block1_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
џ
block1_conv2/Conv1D/SqueezeSqueezeblock1_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0е
block1_conv2/BiasAddBiasAdd$block1_conv2/Conv1D/Squeeze:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @n
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         @\
block1_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ц
block1_pool/ExpandDims
ExpandDimsblock1_conv2/Relu:activations:0#block1_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Ф
block1_pool/MaxPoolMaxPoolblock1_pool/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
Ѕ
block1_pool/SqueezeSqueezeblock1_pool/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
w
block1_dropout/IdentityIdentityblock1_pool/Squeeze:output:0*
T0*+
_output_shapes
:         @m
"block2_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block2_conv1/Conv1D/ExpandDims
ExpandDims block1_dropout/Identity:output:0+block2_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Г
/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0f
$block2_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╚
 block2_conv1/Conv1D/ExpandDims_1
ExpandDims7block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block2_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђн
block2_conv1/Conv1DConv2D'block2_conv1/Conv1D/ExpandDims:output:0)block2_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block2_conv1/Conv1D/SqueezeSqueezeblock2_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block2_conv1/BiasAddBiasAdd$block2_conv1/Conv1D/Squeeze:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђm
"block2_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block2_conv2/Conv1D/ExpandDims
ExpandDimsblock2_conv1/Relu:activations:0+block2_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block2_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block2_conv2/Conv1D/ExpandDims_1
ExpandDims7block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block2_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block2_conv2/Conv1DConv2D'block2_conv2/Conv1D/ExpandDims:output:0)block2_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block2_conv2/Conv1D/SqueezeSqueezeblock2_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block2_conv2/BiasAddBiasAdd$block2_conv2/Conv1D/Squeeze:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ\
block2_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
block2_pool/ExpandDims
ExpandDimsblock2_conv2/Relu:activations:0#block2_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђг
block2_pool/MaxPoolMaxPoolblock2_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
і
block2_pool/SqueezeSqueezeblock2_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
x
block2_dropout/IdentityIdentityblock2_pool/Squeeze:output:0*
T0*,
_output_shapes
:         ђm
"block3_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Х
block3_conv1/Conv1D/ExpandDims
ExpandDims block2_dropout/Identity:output:0+block3_conv1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block3_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block3_conv1/Conv1D/ExpandDims_1
ExpandDims7block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block3_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block3_conv1/Conv1DConv2D'block3_conv1/Conv1D/ExpandDims:output:0)block3_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block3_conv1/Conv1D/SqueezeSqueezeblock3_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block3_conv1/BiasAddBiasAdd$block3_conv1/Conv1D/Squeeze:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђm
"block3_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block3_conv2/Conv1D/ExpandDims
ExpandDimsblock3_conv1/Relu:activations:0+block3_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block3_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block3_conv2/Conv1D/ExpandDims_1
ExpandDims7block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block3_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block3_conv2/Conv1DConv2D'block3_conv2/Conv1D/ExpandDims:output:0)block3_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block3_conv2/Conv1D/SqueezeSqueezeblock3_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block3_conv2/BiasAddBiasAdd$block3_conv2/Conv1D/Squeeze:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ\
block3_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
block3_pool/ExpandDims
ExpandDimsblock3_conv2/Relu:activations:0#block3_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђг
block3_pool/MaxPoolMaxPoolblock3_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
і
block3_pool/SqueezeSqueezeblock3_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Є
flatten_1/ReshapeReshapeblock3_pool/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђЇ
!FeaturesVec/MatMul/ReadVariableOpReadVariableOp*featuresvec_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0Ћ
FeaturesVec/MatMulMatMulflatten_1/Reshape:output:0)FeaturesVec/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @і
"FeaturesVec/BiasAdd/ReadVariableOpReadVariableOp+featuresvec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0џ
FeaturesVec/BiasAddBiasAddFeaturesVec/MatMul:product:0*FeaturesVec/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
IdentityIdentityFeaturesVec/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @¤
NoOpNoOp#^FeaturesVec/BiasAdd/ReadVariableOp"^FeaturesVec/MatMul/ReadVariableOp$^block0_conv1/BiasAdd/ReadVariableOp0^block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block0_conv2/BiasAdd/ReadVariableOp0^block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block1_conv1/BiasAdd/ReadVariableOp0^block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp0^block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp0^block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp0^block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp0^block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp0^block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 2H
"FeaturesVec/BiasAdd/ReadVariableOp"FeaturesVec/BiasAdd/ReadVariableOp2F
!FeaturesVec/MatMul/ReadVariableOp!FeaturesVec/MatMul/ReadVariableOp2J
#block0_conv1/BiasAdd/ReadVariableOp#block0_conv1/BiasAdd/ReadVariableOp2b
/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block0_conv2/BiasAdd/ReadVariableOp#block0_conv2/BiasAdd/ReadVariableOp2b
/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2b
/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2b
/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2b
/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2b
/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2b
/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2b
/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:R N
+
_output_shapes
:         d

_user_specified_nameinput
┘
Ю
L__inference_block2_conv1_layer_call_and_return_conditional_losses_6539620736

inputsB
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Ы
j
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539611735

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         @_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @*ђ
backward_function_namefd__inference___backward_block1_dropout_layer_call_and_return_conditional_losses_6539611731_6539611736:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Г

m
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620932

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Ы
j
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539614573

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:          _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          *ђ
backward_function_namefd__inference___backward_block0_dropout_layer_call_and_return_conditional_losses_6539614568_6539614574:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
╩
O
3__inference_block2_dropout_layer_call_fn_6539620937

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539612059e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
­_
█	
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539618936
input_1-
block0_conv1_6539618827: %
block0_conv1_6539618829: -
block0_conv2_6539618837:  %
block0_conv2_6539618839: -
block1_conv1_6539618852: @%
block1_conv1_6539618854:@-
block1_conv2_6539618862:@@%
block1_conv2_6539618864:@.
block2_conv1_6539618877:@ђ&
block2_conv1_6539618879:	ђ/
block2_conv2_6539618887:ђђ&
block2_conv2_6539618889:	ђ/
block3_conv1_6539618902:ђђ&
block3_conv1_6539618904:	ђ/
block3_conv2_6539618912:ђђ&
block3_conv2_6539618914:	ђ)
featuresvec_6539618928:	ђ@$
featuresvec_6539618930:@
identityѕб#FeaturesVec/StatefulPartitionedCallб$block0_conv1/StatefulPartitionedCallб$block0_conv2/StatefulPartitionedCallб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ы
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskr
subSubstrided_slice:output:0strided_slice_1:output:0*
T0*+
_output_shapes
:         cњ
$block0_conv1/StatefulPartitionedCallStatefulPartitionedCallsub:z:0block0_conv1_6539618827block0_conv1_6539618829*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c: :         c*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539611182И
$block0_conv2/StatefulPartitionedCallStatefulPartitionedCall-block0_conv1/StatefulPartitionedCall:output:0block0_conv2_6539618837block0_conv2_6539618839*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c :  :         c *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539611308└
block0_pool/PartitionedCallPartitionedCall-block0_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:          :          :         c :         c * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539611378ь
block0_dropout/PartitionedCallPartitionedCall$block0_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539611400▓
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall'block0_dropout/PartitionedCall:output:0block1_conv1_6539618852block1_conv1_6539618854*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:          : @:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539611517И
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6539618862block1_conv2_6539618864*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:         @:@@:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539611643└
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:         @:         @:         @:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539611713ь
block1_dropout/PartitionedCallPartitionedCall$block1_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539611735Х
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall'block1_dropout/PartitionedCall:output:0block2_conv1_6539618877block2_conv1_6539618879*
Tin
2*
Tout

2*
_collective_manager_ids
 *Д
_output_shapesћ
Љ:         ђ:         ђ:         ђ:         @:@ђ:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539611852┐
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6539618887block2_conv2_6539618889*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539611978─
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539612048Ь
block2_dropout/PartitionedCallPartitionedCall$block2_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539612070╣
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall'block2_dropout/PartitionedCall:output:0block3_conv1_6539618902block3_conv1_6539618904*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539612187┐
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6539618912block3_conv2_6539618914*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612313─
block3_pool/PartitionedCallPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612383щ
flatten_1/PartitionedCallPartitionedCall$block3_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612427╚
#FeaturesVec/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0featuresvec_6539618928featuresvec_6539618930*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         @:	ђ@:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612485{
IdentityIdentity,FeaturesVec/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ц
NoOpNoOp$^FeaturesVec/StatefulPartitionedCall%^block0_conv1/StatefulPartitionedCall%^block0_conv2/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 2J
#FeaturesVec/StatefulPartitionedCall#FeaturesVec/StatefulPartitionedCall2L
$block0_conv1/StatefulPartitionedCall$block0_conv1/StatefulPartitionedCall2L
$block0_conv2/StatefulPartitionedCall$block0_conv2/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
к
ц
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612485
inputs_01
matmul_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identity
matmul_readvariableop

inputsѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : *}
backward_function_nameca__inference___backward_FeaturesVec_layer_call_and_return_conditional_losses_6539612470_653961248620
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З
Ц
1__inference_block3_conv2_layer_call_fn_6539620810

inputs
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612313t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
нк
П
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620554	
inputN
8block0_conv1_conv1d_expanddims_1_readvariableop_resource: :
,block0_conv1_biasadd_readvariableop_resource: N
8block0_conv2_conv1d_expanddims_1_readvariableop_resource:  :
,block0_conv2_biasadd_readvariableop_resource: N
8block1_conv1_conv1d_expanddims_1_readvariableop_resource: @:
,block1_conv1_biasadd_readvariableop_resource:@N
8block1_conv2_conv1d_expanddims_1_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@O
8block2_conv1_conv1d_expanddims_1_readvariableop_resource:@ђ;
,block2_conv1_biasadd_readvariableop_resource:	ђP
8block2_conv2_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block2_conv2_biasadd_readvariableop_resource:	ђP
8block3_conv1_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block3_conv1_biasadd_readvariableop_resource:	ђP
8block3_conv2_conv1d_expanddims_1_readvariableop_resource:ђђ;
,block3_conv2_biasadd_readvariableop_resource:	ђ=
*featuresvec_matmul_readvariableop_resource:	ђ@9
+featuresvec_biasadd_readvariableop_resource:@
identityѕб"FeaturesVec/BiasAdd/ReadVariableOpб!FeaturesVec/MatMul/ReadVariableOpб#block0_conv1/BiasAdd/ReadVariableOpб/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block0_conv2/BiasAdd/ReadVariableOpб/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block1_conv1/BiasAdd/ReadVariableOpб/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block1_conv2/BiasAdd/ReadVariableOpб/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block2_conv1/BiasAdd/ReadVariableOpб/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block2_conv2/BiasAdd/ReadVariableOpб/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpб#block3_conv1/BiasAdd/ReadVariableOpб/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpб#block3_conv2/BiasAdd/ReadVariableOpб/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         у
strided_sliceStridedSliceinputstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         №
strided_slice_1StridedSliceinputstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskr
subSubstrided_slice:output:0strided_slice_1:output:0*
T0*+
_output_shapes
:         cm
"block0_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ю
block0_conv1/Conv1D/ExpandDims
ExpandDimssub:z:0+block0_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cг
/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block0_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0f
$block0_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block0_conv1/Conv1D/ExpandDims_1
ExpandDims7block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block0_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: М
block0_conv1/Conv1DConv2D'block0_conv1/Conv1D/ExpandDims:output:0)block0_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
џ
block0_conv1/Conv1D/SqueezeSqueezeblock0_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ї
#block0_conv1/BiasAdd/ReadVariableOpReadVariableOp,block0_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
block0_conv1/BiasAddBiasAdd$block0_conv1/Conv1D/Squeeze:output:0+block0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c n
block0_conv1/ReluRelublock0_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         c m
"block0_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ┤
block0_conv2/Conv1D/ExpandDims
ExpandDimsblock0_conv1/Relu:activations:0+block0_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c г
/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block0_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0f
$block0_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block0_conv2/Conv1D/ExpandDims_1
ExpandDims7block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block0_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  М
block0_conv2/Conv1DConv2D'block0_conv2/Conv1D/ExpandDims:output:0)block0_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
џ
block0_conv2/Conv1D/SqueezeSqueezeblock0_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ї
#block0_conv2/BiasAdd/ReadVariableOpReadVariableOp,block0_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
block0_conv2/BiasAddBiasAdd$block0_conv2/Conv1D/Squeeze:output:0+block0_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c n
block0_conv2/ReluRelublock0_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         c \
block0_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ц
block0_pool/ExpandDims
ExpandDimsblock0_conv2/Relu:activations:0#block0_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c Ф
block0_pool/MaxPoolMaxPoolblock0_pool/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
Ѕ
block0_pool/SqueezeSqueezeblock0_pool/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
a
block0_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?ю
block0_dropout/dropout/MulMulblock0_pool/Squeeze:output:0%block0_dropout/dropout/Const:output:0*
T0*+
_output_shapes
:          h
block0_dropout/dropout/ShapeShapeblock0_pool/Squeeze:output:0*
T0*
_output_shapes
:┐
3block0_dropout/dropout/random_uniform/RandomUniformRandomUniform%block0_dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype0*
seed2    j
%block0_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=О
#block0_dropout/dropout/GreaterEqualGreaterEqual<block0_dropout/dropout/random_uniform/RandomUniform:output:0.block0_dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          Љ
block0_dropout/dropout/CastCast'block0_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          џ
block0_dropout/dropout/Mul_1Mulblock0_dropout/dropout/Mul:z:0block0_dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:          m
"block1_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block1_conv1/Conv1D/ExpandDims
ExpandDims block0_dropout/dropout/Mul_1:z:0+block1_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          г
/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0f
$block1_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block1_conv1/Conv1D/ExpandDims_1
ExpandDims7block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block1_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @М
block1_conv1/Conv1DConv2D'block1_conv1/Conv1D/ExpandDims:output:0)block1_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
џ
block1_conv1/Conv1D/SqueezeSqueezeblock1_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0е
block1_conv1/BiasAddBiasAdd$block1_conv1/Conv1D/Squeeze:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @n
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         @m
"block1_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ┤
block1_conv2/Conv1D/ExpandDims
ExpandDimsblock1_conv1/Relu:activations:0+block1_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @г
/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0f
$block1_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : К
 block1_conv2/Conv1D/ExpandDims_1
ExpandDims7block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block1_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@М
block1_conv2/Conv1DConv2D'block1_conv2/Conv1D/ExpandDims:output:0)block1_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
џ
block1_conv2/Conv1D/SqueezeSqueezeblock1_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0е
block1_conv2/BiasAddBiasAdd$block1_conv2/Conv1D/Squeeze:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @n
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         @\
block1_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ц
block1_pool/ExpandDims
ExpandDimsblock1_conv2/Relu:activations:0#block1_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Ф
block1_pool/MaxPoolMaxPoolblock1_pool/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
Ѕ
block1_pool/SqueezeSqueezeblock1_pool/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
a
block1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ю
block1_dropout/dropout/MulMulblock1_pool/Squeeze:output:0%block1_dropout/dropout/Const:output:0*
T0*+
_output_shapes
:         @h
block1_dropout/dropout/ShapeShapeblock1_pool/Squeeze:output:0*
T0*
_output_shapes
:╗
3block1_dropout/dropout/random_uniform/RandomUniformRandomUniform%block1_dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype0*
seed2j
%block1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>О
#block1_dropout/dropout/GreaterEqualGreaterEqual<block1_dropout/dropout/random_uniform/RandomUniform:output:0.block1_dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @Љ
block1_dropout/dropout/CastCast'block1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @џ
block1_dropout/dropout/Mul_1Mulblock1_dropout/dropout/Mul:z:0block1_dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:         @m
"block2_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block2_conv1/Conv1D/ExpandDims
ExpandDims block1_dropout/dropout/Mul_1:z:0+block2_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Г
/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0f
$block2_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╚
 block2_conv1/Conv1D/ExpandDims_1
ExpandDims7block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block2_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђн
block2_conv1/Conv1DConv2D'block2_conv1/Conv1D/ExpandDims:output:0)block2_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block2_conv1/Conv1D/SqueezeSqueezeblock2_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block2_conv1/BiasAddBiasAdd$block2_conv1/Conv1D/Squeeze:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђm
"block2_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block2_conv2/Conv1D/ExpandDims
ExpandDimsblock2_conv1/Relu:activations:0+block2_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block2_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block2_conv2/Conv1D/ExpandDims_1
ExpandDims7block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block2_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block2_conv2/Conv1DConv2D'block2_conv2/Conv1D/ExpandDims:output:0)block2_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block2_conv2/Conv1D/SqueezeSqueezeblock2_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block2_conv2/BiasAddBiasAdd$block2_conv2/Conv1D/Squeeze:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ\
block2_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
block2_pool/ExpandDims
ExpandDimsblock2_conv2/Relu:activations:0#block2_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђг
block2_pool/MaxPoolMaxPoolblock2_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
і
block2_pool/SqueezeSqueezeblock2_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
a
block2_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?Ю
block2_dropout/dropout/MulMulblock2_pool/Squeeze:output:0%block2_dropout/dropout/Const:output:0*
T0*,
_output_shapes
:         ђh
block2_dropout/dropout/ShapeShapeblock2_pool/Squeeze:output:0*
T0*
_output_shapes
:╝
3block2_dropout/dropout/random_uniform/RandomUniformRandomUniform%block2_dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:         ђ*
dtype0*
seed2j
%block2_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>п
#block2_dropout/dropout/GreaterEqualGreaterEqual<block2_dropout/dropout/random_uniform/RandomUniform:output:0.block2_dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ђњ
block2_dropout/dropout/CastCast'block2_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ђЏ
block2_dropout/dropout/Mul_1Mulblock2_dropout/dropout/Mul:z:0block2_dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:         ђm
"block3_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Х
block3_conv1/Conv1D/ExpandDims
ExpandDims block2_dropout/dropout/Mul_1:z:0+block3_conv1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block3_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block3_conv1/Conv1D/ExpandDims_1
ExpandDims7block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block3_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block3_conv1/Conv1DConv2D'block3_conv1/Conv1D/ExpandDims:output:0)block3_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block3_conv1/Conv1D/SqueezeSqueezeblock3_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block3_conv1/BiasAddBiasAdd$block3_conv1/Conv1D/Squeeze:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђm
"block3_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        х
block3_conv2/Conv1D/ExpandDims
ExpandDimsblock3_conv1/Relu:activations:0+block3_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђ«
/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8block3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0f
$block3_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╔
 block3_conv2/Conv1D/ExpandDims_1
ExpandDims7block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0-block3_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђн
block3_conv2/Conv1DConv2D'block3_conv2/Conv1D/ExpandDims:output:0)block3_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Џ
block3_conv2/Conv1D/SqueezeSqueezeblock3_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Е
block3_conv2/BiasAddBiasAdd$block3_conv2/Conv1D/Squeeze:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђo
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ\
block3_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
block3_pool/ExpandDims
ExpandDimsblock3_conv2/Relu:activations:0#block3_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђг
block3_pool/MaxPoolMaxPoolblock3_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
і
block3_pool/SqueezeSqueezeblock3_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Є
flatten_1/ReshapeReshapeblock3_pool/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђЇ
!FeaturesVec/MatMul/ReadVariableOpReadVariableOp*featuresvec_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0Ћ
FeaturesVec/MatMulMatMulflatten_1/Reshape:output:0)FeaturesVec/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @і
"FeaturesVec/BiasAdd/ReadVariableOpReadVariableOp+featuresvec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0џ
FeaturesVec/BiasAddBiasAddFeaturesVec/MatMul:product:0*FeaturesVec/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
IdentityIdentityFeaturesVec/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @¤
NoOpNoOp#^FeaturesVec/BiasAdd/ReadVariableOp"^FeaturesVec/MatMul/ReadVariableOp$^block0_conv1/BiasAdd/ReadVariableOp0^block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block0_conv2/BiasAdd/ReadVariableOp0^block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block1_conv1/BiasAdd/ReadVariableOp0^block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp0^block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp0^block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp0^block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp0^block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp0^block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 2H
"FeaturesVec/BiasAdd/ReadVariableOp"FeaturesVec/BiasAdd/ReadVariableOp2F
!FeaturesVec/MatMul/ReadVariableOp!FeaturesVec/MatMul/ReadVariableOp2J
#block0_conv1/BiasAdd/ReadVariableOp#block0_conv1/BiasAdd/ReadVariableOp2b
/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block0_conv2/BiasAdd/ReadVariableOp#block0_conv2/BiasAdd/ReadVariableOp2b
/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2b
/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2b
/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2b
/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2b
/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2b
/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2b
/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:R N
+
_output_shapes
:         d

_user_specified_nameinput
╬	
љ
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539611713
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block1_pool_layer_call_and_return_conditional_losses_6539611692_6539611714:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
¤
g
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539620865

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Ь
ц
1__inference_block2_conv1_layer_call_fn_6539620720

inputs
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *Д
_output_shapesћ
Љ:         ђ:         ђ:         ђ:         @:@ђ:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539611852t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Я
Є
6__inference_deep_coffea_model_layer_call_fn_6539618815
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@@
	unknown_6:@ 
	unknown_7:@ђ
	unknown_8:	ђ!
	unknown_9:ђђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ"

unknown_13:ђђ

unknown_14:	ђ

unknown_15:	ђ@

unknown_16:@
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*S
ToutK
I2G*
_collective_manager_ids
 *Ч
_output_shapesж
Т:         @:	ђ@:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ: :         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         @:@ђ:         @:         @:         @:         @: :         @:         @:         @:         @:         @:         @:@@:         @:         @:         @:          : @:          :          :          :          : :          :         c :         c :         c :         c :         c :  :         c :         c :         c :         c: :         c:         c:         c:         d*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539618594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
ЉФ
ѕ!
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539618594
input_0-
block0_conv1_6539616245: %
block0_conv1_6539616247: -
block0_conv2_6539616255:  %
block0_conv2_6539616257: -
block1_conv1_6539616384: @%
block1_conv1_6539616386:@-
block1_conv2_6539616394:@@%
block1_conv2_6539616396:@.
block2_conv1_6539616523:@ђ&
block2_conv1_6539616525:	ђ/
block2_conv2_6539616533:ђђ&
block2_conv2_6539616535:	ђ/
block3_conv1_6539616662:ђђ&
block3_conv1_6539616664:	ђ/
block3_conv2_6539616672:ђђ&
block3_conv2_6539616674:	ђ)
featuresvec_6539616688:	ђ@$
featuresvec_6539616690:@
identity'
#featuresvec_statefulpartitionedcall)
%featuresvec_statefulpartitionedcall_0
flatten_1_partitionedcall
block3_pool_partitionedcall!
block3_pool_partitionedcall_0!
block3_pool_partitionedcall_1(
$block3_conv2_statefulpartitionedcall*
&block3_conv2_statefulpartitionedcall_0*
&block3_conv2_statefulpartitionedcall_1*
&block3_conv2_statefulpartitionedcall_2*
&block3_conv2_statefulpartitionedcall_3(
$block3_conv1_statefulpartitionedcall*
&block3_conv1_statefulpartitionedcall_0*
&block3_conv1_statefulpartitionedcall_1*
&block3_conv1_statefulpartitionedcall_2*
&block3_conv1_statefulpartitionedcall_3*
&block2_dropout_statefulpartitionedcall,
(block2_dropout_statefulpartitionedcall_0,
(block2_dropout_statefulpartitionedcall_1,
(block2_dropout_statefulpartitionedcall_2
block2_pool_partitionedcall!
block2_pool_partitionedcall_0!
block2_pool_partitionedcall_1(
$block2_conv2_statefulpartitionedcall*
&block2_conv2_statefulpartitionedcall_0*
&block2_conv2_statefulpartitionedcall_1*
&block2_conv2_statefulpartitionedcall_2*
&block2_conv2_statefulpartitionedcall_3(
$block2_conv1_statefulpartitionedcall*
&block2_conv1_statefulpartitionedcall_0*
&block2_conv1_statefulpartitionedcall_1*
&block2_conv1_statefulpartitionedcall_2*
&block2_conv1_statefulpartitionedcall_3*
&block1_dropout_statefulpartitionedcall,
(block1_dropout_statefulpartitionedcall_0,
(block1_dropout_statefulpartitionedcall_1,
(block1_dropout_statefulpartitionedcall_2
block1_pool_partitionedcall!
block1_pool_partitionedcall_0!
block1_pool_partitionedcall_1(
$block1_conv2_statefulpartitionedcall*
&block1_conv2_statefulpartitionedcall_0*
&block1_conv2_statefulpartitionedcall_1*
&block1_conv2_statefulpartitionedcall_2*
&block1_conv2_statefulpartitionedcall_3(
$block1_conv1_statefulpartitionedcall*
&block1_conv1_statefulpartitionedcall_0*
&block1_conv1_statefulpartitionedcall_1*
&block1_conv1_statefulpartitionedcall_2*
&block1_conv1_statefulpartitionedcall_3*
&block0_dropout_statefulpartitionedcall,
(block0_dropout_statefulpartitionedcall_0,
(block0_dropout_statefulpartitionedcall_1,
(block0_dropout_statefulpartitionedcall_2
block0_pool_partitionedcall!
block0_pool_partitionedcall_0!
block0_pool_partitionedcall_1(
$block0_conv2_statefulpartitionedcall*
&block0_conv2_statefulpartitionedcall_0*
&block0_conv2_statefulpartitionedcall_1*
&block0_conv2_statefulpartitionedcall_2*
&block0_conv2_statefulpartitionedcall_3(
$block0_conv1_statefulpartitionedcall*
&block0_conv1_statefulpartitionedcall_0*
&block0_conv1_statefulpartitionedcall_1*
&block0_conv1_statefulpartitionedcall_2*
&block0_conv1_statefulpartitionedcall_3
strided_slice
strided_slice_1	
inputѕб#FeaturesVec/StatefulPartitionedCallб$block0_conv1/StatefulPartitionedCallб$block0_conv2/StatefulPartitionedCallб&block0_dropout/StatefulPartitionedCallб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб&block1_dropout/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб&block2_dropout/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Й
strided_slice_0StridedSliceinput_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         к
strided_slice_1_0StridedSliceinput_0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*

begin_mask*
end_maskv
subSubstrided_slice_0:output:0strided_slice_1_0:output:0*
T0*+
_output_shapes
:         cњ
$block0_conv1/StatefulPartitionedCallStatefulPartitionedCallsub:z:0block0_conv1_6539616245block0_conv1_6539616247*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c: :         c*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539614995И
$block0_conv2/StatefulPartitionedCallStatefulPartitionedCall-block0_conv1/StatefulPartitionedCall:output:0block0_conv2_6539616255block0_conv2_6539616257*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c :  :         c *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539614747└
block0_pool/PartitionedCallPartitionedCall-block0_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:          :          :         c :         c * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539614616╚
&block0_dropout/StatefulPartitionedCallStatefulPartitionedCall$block0_pool/PartitionedCall:output:0*
Tin
2*
Tout	
2*
_collective_manager_ids
 *r
_output_shapes`
^:          :          :          :          : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539617521║
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall/block0_dropout/StatefulPartitionedCall:output:0block1_conv1_6539616384block1_conv1_6539616386*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:          : @:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539614369И
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6539616394block1_conv2_6539616396*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:         @:@@:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539614121└
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:         @:         @:         @:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539613990ы
&block1_dropout/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0'^block0_dropout/StatefulPartitionedCall*
Tin
2*
Tout	
2*
_collective_manager_ids
 *r
_output_shapes`
^:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539617166Й
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall/block1_dropout/StatefulPartitionedCall:output:0block2_conv1_6539616523block2_conv1_6539616525*
Tin
2*
Tout

2*
_collective_manager_ids
 *Д
_output_shapesћ
Љ:         ђ:         ђ:         ђ:         @:@ђ:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539613743┐
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6539616533block2_conv2_6539616535*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539613495─
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539613364ш
&block2_dropout/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0'^block1_dropout/StatefulPartitionedCall*
Tin
2*
Tout	
2*
_collective_manager_ids
 *v
_output_shapesd
b:         ђ:         ђ:         ђ:         ђ: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539616811┴
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall/block2_dropout/StatefulPartitionedCall:output:0block3_conv1_6539616662block3_conv1_6539616664*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539613117┐
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6539616672block3_conv2_6539616674*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612869─
block3_pool/PartitionedCallPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612738щ
flatten_1/PartitionedCallPartitionedCall$block3_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612673╚
#FeaturesVec/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0featuresvec_6539616688featuresvec_6539616690*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         @:	ђ@:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612522{
IdentityIdentity,FeaturesVec/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @Ъ
NoOpNoOp$^FeaturesVec/StatefulPartitionedCall%^block0_conv1/StatefulPartitionedCall%^block0_conv2/StatefulPartitionedCall'^block0_dropout/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall'^block1_dropout/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall'^block2_dropout/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "U
$block0_conv1_statefulpartitionedcall-block0_conv1/StatefulPartitionedCall:output:1"W
&block0_conv1_statefulpartitionedcall_0-block0_conv1/StatefulPartitionedCall:output:2"W
&block0_conv1_statefulpartitionedcall_1-block0_conv1/StatefulPartitionedCall:output:3"W
&block0_conv1_statefulpartitionedcall_2-block0_conv1/StatefulPartitionedCall:output:4"W
&block0_conv1_statefulpartitionedcall_3-block0_conv1/StatefulPartitionedCall:output:5"U
$block0_conv2_statefulpartitionedcall-block0_conv2/StatefulPartitionedCall:output:1"W
&block0_conv2_statefulpartitionedcall_0-block0_conv2/StatefulPartitionedCall:output:2"W
&block0_conv2_statefulpartitionedcall_1-block0_conv2/StatefulPartitionedCall:output:3"W
&block0_conv2_statefulpartitionedcall_2-block0_conv2/StatefulPartitionedCall:output:4"W
&block0_conv2_statefulpartitionedcall_3-block0_conv2/StatefulPartitionedCall:output:5"Y
&block0_dropout_statefulpartitionedcall/block0_dropout/StatefulPartitionedCall:output:1"[
(block0_dropout_statefulpartitionedcall_0/block0_dropout/StatefulPartitionedCall:output:2"[
(block0_dropout_statefulpartitionedcall_1/block0_dropout/StatefulPartitionedCall:output:3"[
(block0_dropout_statefulpartitionedcall_2/block0_dropout/StatefulPartitionedCall:output:4"C
block0_pool_partitionedcall$block0_pool/PartitionedCall:output:1"E
block0_pool_partitionedcall_0$block0_pool/PartitionedCall:output:2"E
block0_pool_partitionedcall_1$block0_pool/PartitionedCall:output:3"U
$block1_conv1_statefulpartitionedcall-block1_conv1/StatefulPartitionedCall:output:1"W
&block1_conv1_statefulpartitionedcall_0-block1_conv1/StatefulPartitionedCall:output:2"W
&block1_conv1_statefulpartitionedcall_1-block1_conv1/StatefulPartitionedCall:output:3"W
&block1_conv1_statefulpartitionedcall_2-block1_conv1/StatefulPartitionedCall:output:4"W
&block1_conv1_statefulpartitionedcall_3-block1_conv1/StatefulPartitionedCall:output:5"U
$block1_conv2_statefulpartitionedcall-block1_conv2/StatefulPartitionedCall:output:1"W
&block1_conv2_statefulpartitionedcall_0-block1_conv2/StatefulPartitionedCall:output:2"W
&block1_conv2_statefulpartitionedcall_1-block1_conv2/StatefulPartitionedCall:output:3"W
&block1_conv2_statefulpartitionedcall_2-block1_conv2/StatefulPartitionedCall:output:4"W
&block1_conv2_statefulpartitionedcall_3-block1_conv2/StatefulPartitionedCall:output:5"Y
&block1_dropout_statefulpartitionedcall/block1_dropout/StatefulPartitionedCall:output:1"[
(block1_dropout_statefulpartitionedcall_0/block1_dropout/StatefulPartitionedCall:output:2"[
(block1_dropout_statefulpartitionedcall_1/block1_dropout/StatefulPartitionedCall:output:3"[
(block1_dropout_statefulpartitionedcall_2/block1_dropout/StatefulPartitionedCall:output:4"C
block1_pool_partitionedcall$block1_pool/PartitionedCall:output:1"E
block1_pool_partitionedcall_0$block1_pool/PartitionedCall:output:2"E
block1_pool_partitionedcall_1$block1_pool/PartitionedCall:output:3"U
$block2_conv1_statefulpartitionedcall-block2_conv1/StatefulPartitionedCall:output:1"W
&block2_conv1_statefulpartitionedcall_0-block2_conv1/StatefulPartitionedCall:output:2"W
&block2_conv1_statefulpartitionedcall_1-block2_conv1/StatefulPartitionedCall:output:3"W
&block2_conv1_statefulpartitionedcall_2-block2_conv1/StatefulPartitionedCall:output:4"W
&block2_conv1_statefulpartitionedcall_3-block2_conv1/StatefulPartitionedCall:output:5"U
$block2_conv2_statefulpartitionedcall-block2_conv2/StatefulPartitionedCall:output:1"W
&block2_conv2_statefulpartitionedcall_0-block2_conv2/StatefulPartitionedCall:output:2"W
&block2_conv2_statefulpartitionedcall_1-block2_conv2/StatefulPartitionedCall:output:3"W
&block2_conv2_statefulpartitionedcall_2-block2_conv2/StatefulPartitionedCall:output:4"W
&block2_conv2_statefulpartitionedcall_3-block2_conv2/StatefulPartitionedCall:output:5"Y
&block2_dropout_statefulpartitionedcall/block2_dropout/StatefulPartitionedCall:output:1"[
(block2_dropout_statefulpartitionedcall_0/block2_dropout/StatefulPartitionedCall:output:2"[
(block2_dropout_statefulpartitionedcall_1/block2_dropout/StatefulPartitionedCall:output:3"[
(block2_dropout_statefulpartitionedcall_2/block2_dropout/StatefulPartitionedCall:output:4"C
block2_pool_partitionedcall$block2_pool/PartitionedCall:output:1"E
block2_pool_partitionedcall_0$block2_pool/PartitionedCall:output:2"E
block2_pool_partitionedcall_1$block2_pool/PartitionedCall:output:3"U
$block3_conv1_statefulpartitionedcall-block3_conv1/StatefulPartitionedCall:output:1"W
&block3_conv1_statefulpartitionedcall_0-block3_conv1/StatefulPartitionedCall:output:2"W
&block3_conv1_statefulpartitionedcall_1-block3_conv1/StatefulPartitionedCall:output:3"W
&block3_conv1_statefulpartitionedcall_2-block3_conv1/StatefulPartitionedCall:output:4"W
&block3_conv1_statefulpartitionedcall_3-block3_conv1/StatefulPartitionedCall:output:5"U
$block3_conv2_statefulpartitionedcall-block3_conv2/StatefulPartitionedCall:output:1"W
&block3_conv2_statefulpartitionedcall_0-block3_conv2/StatefulPartitionedCall:output:2"W
&block3_conv2_statefulpartitionedcall_1-block3_conv2/StatefulPartitionedCall:output:3"W
&block3_conv2_statefulpartitionedcall_2-block3_conv2/StatefulPartitionedCall:output:4"W
&block3_conv2_statefulpartitionedcall_3-block3_conv2/StatefulPartitionedCall:output:5"C
block3_pool_partitionedcall$block3_pool/PartitionedCall:output:1"E
block3_pool_partitionedcall_0$block3_pool/PartitionedCall:output:2"E
block3_pool_partitionedcall_1$block3_pool/PartitionedCall:output:3"S
#featuresvec_statefulpartitionedcall,FeaturesVec/StatefulPartitionedCall:output:1"U
%featuresvec_statefulpartitionedcall_0,FeaturesVec/StatefulPartitionedCall:output:2"?
flatten_1_partitionedcall"flatten_1/PartitionedCall:output:1"
identityIdentity:output:0"
inputinput_0")
strided_slicestrided_slice_0:output:0"-
strided_slice_1strided_slice_1_0:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : *Ѓ
backward_function_nameig__inference___backward_deep_coffea_model_layer_call_and_return_conditional_losses_6539618225_65396185952J
#FeaturesVec/StatefulPartitionedCall#FeaturesVec/StatefulPartitionedCall2L
$block0_conv1/StatefulPartitionedCall$block0_conv1/StatefulPartitionedCall2L
$block0_conv2/StatefulPartitionedCall$block0_conv2/StatefulPartitionedCall2P
&block0_dropout/StatefulPartitionedCall&block0_dropout/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2P
&block1_dropout/StatefulPartitionedCall&block1_dropout/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2P
&block2_dropout/StatefulPartitionedCall&block2_dropout/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall:R N
+
_output_shapes
:         d

_user_specified_nameinput
ѕ
L
0__inference_block3_pool_layer_call_fn_6539620870

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *T
fORM
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539611049v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
¤
Џ
L__inference_block0_conv2_layer_call_and_return_conditional_losses_6539620646

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c 
 
_user_specified_nameinputs
Ш
Ё
6__inference_deep_coffea_model_layer_call_fn_6539620150	
input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@@
	unknown_6:@ 
	unknown_7:@ђ
	unknown_8:	ђ!
	unknown_9:ђђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ"

unknown_13:ђђ

unknown_14:	ђ

unknown_15:	ђ@

unknown_16:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*G
Tout?
=2;*
_collective_manager_ids
 *ц
_output_shapesЉ
ј:         @:	ђ@:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         @:@ђ:         @:         @:         @:         @:         @:         @:         @:@@:         @:         @:         @:          : @:          :          :         c :         c :         c :         c :         c :  :         c :         c :         c :         c: :         c:         c:         c:         d*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539615827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d

_user_specified_nameinput
Ш
»
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539616516
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?j
dropout/MulMulinputs_0dropout/Const:output:0*
T0*+
_output_shapes
:         @E
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @" 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @*ђ
backward_function_namefd__inference___backward_block1_dropout_layer_call_and_return_conditional_losses_6539616478_6539616517:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
х

m
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620959

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:б
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ђ*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ђt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ђn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ђ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
»
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539616377
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?j
dropout/MulMulinputs_0dropout/Const:output:0*
T0*+
_output_shapes
:          E
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          " 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime**
_input_shapes
:          *ђ
backward_function_namefd__inference___backward_block0_dropout_layer_call_and_return_conditional_losses_6539616339_6539616378:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
ы
l
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620920

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         @_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
¤
g
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539620852

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
н
ь
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539614995
inputs_0A
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cњ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c: : *~
backward_function_namedb__inference___backward_block0_conv1_layer_call_and_return_conditional_losses_6539614956_653961499620
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c
 
_user_specified_nameinputs
▀
ъ
L__inference_block2_conv2_layer_call_and_return_conditional_losses_6539620766

inputsC
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ѓ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
м	
§
K__inference_FeaturesVec_layer_call_and_return_conditional_losses_6539620586

inputs1
matmul_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┼
e
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539612395

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬	
љ
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539611378
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block0_pool_layer_call_and_return_conditional_losses_6539611357_6539611379:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
н
ь
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539611308
inputs_0A
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c : : *~
backward_function_namedb__inference___backward_block0_conv2_layer_call_and_return_conditional_losses_6539611272_653961130920
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c 
 
_user_specified_nameinputs
¤
g
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539611004

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Г

m
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539616099

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
■
»
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539616655
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?k
dropout/MulMulinputs_0dropout/Const:output:0*
T0*,
_output_shapes
:         ђE
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:б
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ђ*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ђt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ђn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ђ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ђ" 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*ђ
backward_function_namefd__inference___backward_block2_dropout_layer_call_and_return_conditional_losses_6539616617_6539616656:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ѕ
L
0__inference_block0_pool_layer_call_fn_6539620831

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *T
fORM
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539611004v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
С
­
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539612187
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block3_conv1_layer_call_and_return_conditional_losses_6539612151_653961218820
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ђ1
╚

#__inference__traced_save_6539621036
file_prefixD
@savev2_deep_coffea_model_block0_conv1_kernel_read_readvariableopB
>savev2_deep_coffea_model_block0_conv1_bias_read_readvariableopD
@savev2_deep_coffea_model_block0_conv2_kernel_read_readvariableopB
>savev2_deep_coffea_model_block0_conv2_bias_read_readvariableopD
@savev2_deep_coffea_model_block1_conv1_kernel_read_readvariableopB
>savev2_deep_coffea_model_block1_conv1_bias_read_readvariableopD
@savev2_deep_coffea_model_block1_conv2_kernel_read_readvariableopB
>savev2_deep_coffea_model_block1_conv2_bias_read_readvariableopD
@savev2_deep_coffea_model_block2_conv1_kernel_read_readvariableopB
>savev2_deep_coffea_model_block2_conv1_bias_read_readvariableopD
@savev2_deep_coffea_model_block2_conv2_kernel_read_readvariableopB
>savev2_deep_coffea_model_block2_conv2_bias_read_readvariableopD
@savev2_deep_coffea_model_block3_conv1_kernel_read_readvariableopB
>savev2_deep_coffea_model_block3_conv1_bias_read_readvariableopD
@savev2_deep_coffea_model_block3_conv2_kernel_read_readvariableopB
>savev2_deep_coffea_model_block3_conv2_bias_read_readvariableopC
?savev2_deep_coffea_model_featuresvec_kernel_read_readvariableopA
=savev2_deep_coffea_model_featuresvec_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Р
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueЂB■B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЊ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B м

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_deep_coffea_model_block0_conv1_kernel_read_readvariableop>savev2_deep_coffea_model_block0_conv1_bias_read_readvariableop@savev2_deep_coffea_model_block0_conv2_kernel_read_readvariableop>savev2_deep_coffea_model_block0_conv2_bias_read_readvariableop@savev2_deep_coffea_model_block1_conv1_kernel_read_readvariableop>savev2_deep_coffea_model_block1_conv1_bias_read_readvariableop@savev2_deep_coffea_model_block1_conv2_kernel_read_readvariableop>savev2_deep_coffea_model_block1_conv2_bias_read_readvariableop@savev2_deep_coffea_model_block2_conv1_kernel_read_readvariableop>savev2_deep_coffea_model_block2_conv1_bias_read_readvariableop@savev2_deep_coffea_model_block2_conv2_kernel_read_readvariableop>savev2_deep_coffea_model_block2_conv2_bias_read_readvariableop@savev2_deep_coffea_model_block3_conv1_kernel_read_readvariableop>savev2_deep_coffea_model_block3_conv1_bias_read_readvariableop@savev2_deep_coffea_model_block3_conv2_kernel_read_readvariableop>savev2_deep_coffea_model_block3_conv2_bias_read_readvariableop?savev2_deep_coffea_model_featuresvec_kernel_read_readvariableop=savev2_deep_coffea_model_featuresvec_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Н
_input_shapes├
└: : : :  : : @:@:@@:@:@ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:	ђ@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:)	%
#
_output_shapes
:@ђ:!


_output_shapes	
:ђ:*&
$
_output_shapes
:ђђ:!

_output_shapes	
:ђ:*&
$
_output_shapes
:ђђ:!

_output_shapes	
:ђ:*&
$
_output_shapes
:ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ@: 

_output_shapes
:@:

_output_shapes
: 
С
­
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612313
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block3_conv2_layer_call_and_return_conditional_losses_6539612277_653961231420
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы
l
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539611389

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:          _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
¤
g
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539611019

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ш
l
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620947

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ђ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
»
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539617521
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?j
dropout/MulMulinputs_0dropout/Const:output:0*
T0*+
_output_shapes
:          E
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          " 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime**
_input_shapes
:          *ђ
backward_function_namefd__inference___backward_block0_dropout_layer_call_and_return_conditional_losses_6539617482_6539617522:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
я
№
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539613743
inputs_0B
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : *~
backward_function_namedb__inference___backward_block2_conv1_layer_call_and_return_conditional_losses_6539613704_653961374420
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
■
щ
(__inference_signature_wrapper_6539620051
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@@
	unknown_6:@ 
	unknown_7:@ђ
	unknown_8:	ђ!
	unknown_9:ђђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ"

unknown_13:ђђ

unknown_14:	ђ

unknown_15:	ђ@

unknown_16:@
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*?
Tout7
523*
_collective_manager_ids
 *У	
_output_shapesН	
м	:         @:	ђ@:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         @:@ђ:         @:         @:         @:         @:         @:         @:@@:         @:         @:          : @:          :          :         c :         c :         c :         c :  :         c :         c :         c: :         c:         c:         c:         d*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__forward__wrapped_model_6539619961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
Ш
j
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539613321

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ђ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*ђ
backward_function_namefd__inference___backward_block2_dropout_layer_call_and_return_conditional_losses_6539613316_6539613322:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
к
ц
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612522
inputs_01
matmul_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identity
matmul_readvariableop

inputsѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : *}
backward_function_nameca__inference___backward_FeaturesVec_layer_call_and_return_conditional_losses_6539612504_653961252320
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
g
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539611049

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╬	
љ
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539613990
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block1_pool_layer_call_and_return_conditional_losses_6539613968_6539613991:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
С
q
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612673
inputs_0
identity

inputsV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       _
ReshapeReshapeinputs_0Const:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*{
backward_function_namea___inference___backward_flatten_1_layer_call_and_return_conditional_losses_6539612662_6539612674:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
я
№
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539611852
inputs_0B
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : *~
backward_function_namedb__inference___backward_block2_conv1_layer_call_and_return_conditional_losses_6539611816_653961185320
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ы
l
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620893

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:          _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Ш
j
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539612070

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ђ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*ђ
backward_function_namefd__inference___backward_block2_dropout_layer_call_and_return_conditional_losses_6539612066_6539612071:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
┌f
о

Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539619069
input_1-
block0_conv1_6539618948: %
block0_conv1_6539618950: -
block0_conv2_6539618958:  %
block0_conv2_6539618960: -
block1_conv1_6539618977: @%
block1_conv1_6539618979:@-
block1_conv2_6539618987:@@%
block1_conv2_6539618989:@.
block2_conv1_6539619006:@ђ&
block2_conv1_6539619008:	ђ/
block2_conv2_6539619016:ђђ&
block2_conv2_6539619018:	ђ/
block3_conv1_6539619035:ђђ&
block3_conv1_6539619037:	ђ/
block3_conv2_6539619045:ђђ&
block3_conv2_6539619047:	ђ)
featuresvec_6539619061:	ђ@$
featuresvec_6539619063:@
identityѕб#FeaturesVec/StatefulPartitionedCallб$block0_conv1/StatefulPartitionedCallб$block0_conv2/StatefulPartitionedCallб&block0_dropout/StatefulPartitionedCallб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб&block1_dropout/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб&block2_dropout/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ы
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskr
subSubstrided_slice:output:0strided_slice_1:output:0*
T0*+
_output_shapes
:         cњ
$block0_conv1/StatefulPartitionedCallStatefulPartitionedCallsub:z:0block0_conv1_6539618948block0_conv1_6539618950*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c: :         c*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539611182И
$block0_conv2/StatefulPartitionedCallStatefulPartitionedCall-block0_conv1/StatefulPartitionedCall:output:0block0_conv2_6539618958block0_conv2_6539618960*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c :  :         c *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539611308└
block0_pool/PartitionedCallPartitionedCall-block0_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:          :          :         c :         c * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539611378╚
&block0_dropout/StatefulPartitionedCallStatefulPartitionedCall$block0_pool/PartitionedCall:output:0*
Tin
2*
Tout	
2*
_collective_manager_ids
 *r
_output_shapes`
^:          :          :          :          : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539616377║
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall/block0_dropout/StatefulPartitionedCall:output:0block1_conv1_6539618977block1_conv1_6539618979*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:          : @:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539611517И
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6539618987block1_conv2_6539618989*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:         @:@@:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539611643└
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:         @:         @:         @:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539611713ы
&block1_dropout/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0'^block0_dropout/StatefulPartitionedCall*
Tin
2*
Tout	
2*
_collective_manager_ids
 *r
_output_shapes`
^:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539616516Й
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall/block1_dropout/StatefulPartitionedCall:output:0block2_conv1_6539619006block2_conv1_6539619008*
Tin
2*
Tout

2*
_collective_manager_ids
 *Д
_output_shapesћ
Љ:         ђ:         ђ:         ђ:         @:@ђ:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539611852┐
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6539619016block2_conv2_6539619018*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539611978─
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539612048ш
&block2_dropout/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0'^block1_dropout/StatefulPartitionedCall*
Tin
2*
Tout	
2*
_collective_manager_ids
 *v
_output_shapesd
b:         ђ:         ђ:         ђ:         ђ: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539616655┴
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall/block2_dropout/StatefulPartitionedCall:output:0block3_conv1_6539619035block3_conv1_6539619037*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539612187┐
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6539619045block3_conv2_6539619047*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612313─
block3_pool/PartitionedCallPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612383щ
flatten_1/PartitionedCallPartitionedCall$block3_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612427╚
#FeaturesVec/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0featuresvec_6539619061featuresvec_6539619063*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         @:	ђ@:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612485{
IdentityIdentity,FeaturesVec/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @Ъ
NoOpNoOp$^FeaturesVec/StatefulPartitionedCall%^block0_conv1/StatefulPartitionedCall%^block0_conv2/StatefulPartitionedCall'^block0_dropout/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall'^block1_dropout/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall'^block2_dropout/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 2J
#FeaturesVec/StatefulPartitionedCall#FeaturesVec/StatefulPartitionedCall2L
$block0_conv1/StatefulPartitionedCall$block0_conv1/StatefulPartitionedCall2L
$block0_conv2/StatefulPartitionedCall$block0_conv2/StatefulPartitionedCall2P
&block0_dropout/StatefulPartitionedCall&block0_dropout/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2P
&block1_dropout/StatefulPartitionedCall&block1_dropout/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2P
&block2_dropout/StatefulPartitionedCall&block2_dropout/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
у
б
1__inference_block1_conv2_layer_call_fn_6539620690

inputs
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:         @:@@:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539611643s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
э
ъ
0__inference_FeaturesVec_layer_call_fn_6539620576

inputs
unknown:	ђ@
	unknown_0:@
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         @:	ђ@:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
н
ь
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539614369
inputs_0A
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : *~
backward_function_namedb__inference___backward_block1_conv1_layer_call_and_return_conditional_losses_6539614330_653961437020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Ы
j
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539611400

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:          _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          *ђ
backward_function_namefd__inference___backward_block0_dropout_layer_call_and_return_conditional_losses_6539611396_6539611401:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Ч
Є
6__inference_deep_coffea_model_layer_call_fn_6539615925
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@@
	unknown_6:@ 
	unknown_7:@ђ
	unknown_8:	ђ!
	unknown_9:ђђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ"

unknown_13:ђђ

unknown_14:	ђ

unknown_15:	ђ@

unknown_16:@
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*G
Tout?
=2;*
_collective_manager_ids
 *ц
_output_shapesЉ
ј:         @:	ђ@:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         @:@ђ:         @:         @:         @:         @:         @:         @:         @:@@:         @:         @:         @:          : @:          :          :         c :         c :         c :         c :         c :  :         c :         c :         c :         c: :         c:         c:         c:         d*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539615827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
З
Ц
1__inference_block3_conv1_layer_call_fn_6539620780

inputs
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539612187t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Г

m
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539616046

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╬	
љ
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612738
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block3_pool_layer_call_and_return_conditional_losses_6539612716_6539612739:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
у
б
1__inference_block0_conv1_layer_call_fn_6539620600

inputs
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c: :         c*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539611182s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         c `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         c
 
_user_specified_nameinputs
С▄
╗
%__inference__wrapped_model_6539610992
input_1`
Jdeep_coffea_model_block0_conv1_conv1d_expanddims_1_readvariableop_resource: L
>deep_coffea_model_block0_conv1_biasadd_readvariableop_resource: `
Jdeep_coffea_model_block0_conv2_conv1d_expanddims_1_readvariableop_resource:  L
>deep_coffea_model_block0_conv2_biasadd_readvariableop_resource: `
Jdeep_coffea_model_block1_conv1_conv1d_expanddims_1_readvariableop_resource: @L
>deep_coffea_model_block1_conv1_biasadd_readvariableop_resource:@`
Jdeep_coffea_model_block1_conv2_conv1d_expanddims_1_readvariableop_resource:@@L
>deep_coffea_model_block1_conv2_biasadd_readvariableop_resource:@a
Jdeep_coffea_model_block2_conv1_conv1d_expanddims_1_readvariableop_resource:@ђM
>deep_coffea_model_block2_conv1_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block2_conv2_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block2_conv2_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block3_conv1_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block3_conv1_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block3_conv2_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block3_conv2_biasadd_readvariableop_resource:	ђO
<deep_coffea_model_featuresvec_matmul_readvariableop_resource:	ђ@K
=deep_coffea_model_featuresvec_biasadd_readvariableop_resource:@
identityѕб4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOpб3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOpб5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpz
%deep_coffea_model/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           |
'deep_coffea_model/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            |
'deep_coffea_model/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ▒
deep_coffea_model/strided_sliceStridedSliceinput_1.deep_coffea_model/strided_slice/stack:output:00deep_coffea_model/strided_slice/stack_1:output:00deep_coffea_model/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_mask|
'deep_coffea_model/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)deep_coffea_model/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            ~
)deep_coffea_model/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╣
!deep_coffea_model/strided_slice_1StridedSliceinput_10deep_coffea_model/strided_slice_1/stack:output:02deep_coffea_model/strided_slice_1/stack_1:output:02deep_coffea_model/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskе
deep_coffea_model/subSub(deep_coffea_model/strided_slice:output:0*deep_coffea_model/strided_slice_1:output:0*
T0*+
_output_shapes
:         c
4deep_coffea_model/block0_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        м
0deep_coffea_model/block0_conv1/Conv1D/ExpandDims
ExpandDimsdeep_coffea_model/sub:z:0=deep_coffea_model/block0_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cл
Adeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block0_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0x
6deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ѕ
%deep_coffea_model/block0_conv1/Conv1DConv2D9deep_coffea_model/block0_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
Й
-deep_coffea_model/block0_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block0_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ░
5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block0_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0я
&deep_coffea_model/block0_conv1/BiasAddBiasAdd6deep_coffea_model/block0_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c њ
#deep_coffea_model/block0_conv1/ReluRelu/deep_coffea_model/block0_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         c 
4deep_coffea_model/block0_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ж
0deep_coffea_model/block0_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block0_conv1/Relu:activations:0=deep_coffea_model/block0_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c л
Adeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block0_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0x
6deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ѕ
%deep_coffea_model/block0_conv2/Conv1DConv2D9deep_coffea_model/block0_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
Й
-deep_coffea_model/block0_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block0_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ░
5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block0_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0я
&deep_coffea_model/block0_conv2/BiasAddBiasAdd6deep_coffea_model/block0_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c њ
#deep_coffea_model/block0_conv2/ReluRelu/deep_coffea_model/block0_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         c n
,deep_coffea_model/block0_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :┌
(deep_coffea_model/block0_pool/ExpandDims
ExpandDims1deep_coffea_model/block0_conv2/Relu:activations:05deep_coffea_model/block0_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c ¤
%deep_coffea_model/block0_pool/MaxPoolMaxPool1deep_coffea_model/block0_pool/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
Г
%deep_coffea_model/block0_pool/SqueezeSqueeze.deep_coffea_model/block0_pool/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
Џ
)deep_coffea_model/block0_dropout/IdentityIdentity.deep_coffea_model/block0_pool/Squeeze:output:0*
T0*+
_output_shapes
:          
4deep_coffea_model/block1_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block1_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block0_dropout/Identity:output:0=deep_coffea_model/block1_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          л
Adeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0x
6deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @Ѕ
%deep_coffea_model/block1_conv1/Conv1DConv2D9deep_coffea_model/block1_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Й
-deep_coffea_model/block1_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block1_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ░
5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
&deep_coffea_model/block1_conv1/BiasAddBiasAdd6deep_coffea_model/block1_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @њ
#deep_coffea_model/block1_conv1/ReluRelu/deep_coffea_model/block1_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         @
4deep_coffea_model/block1_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ж
0deep_coffea_model/block1_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block1_conv1/Relu:activations:0=deep_coffea_model/block1_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @л
Adeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0x
6deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ѕ
%deep_coffea_model/block1_conv2/Conv1DConv2D9deep_coffea_model/block1_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Й
-deep_coffea_model/block1_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block1_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ░
5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
&deep_coffea_model/block1_conv2/BiasAddBiasAdd6deep_coffea_model/block1_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @њ
#deep_coffea_model/block1_conv2/ReluRelu/deep_coffea_model/block1_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         @n
,deep_coffea_model/block1_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :┌
(deep_coffea_model/block1_pool/ExpandDims
ExpandDims1deep_coffea_model/block1_conv2/Relu:activations:05deep_coffea_model/block1_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @¤
%deep_coffea_model/block1_pool/MaxPoolMaxPool1deep_coffea_model/block1_pool/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
Г
%deep_coffea_model/block1_pool/SqueezeSqueeze.deep_coffea_model/block1_pool/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
Џ
)deep_coffea_model/block1_dropout/IdentityIdentity.deep_coffea_model/block1_pool/Squeeze:output:0*
T0*+
_output_shapes
:         @
4deep_coffea_model/block2_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block2_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block1_dropout/Identity:output:0=deep_coffea_model/block2_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Л
Adeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0x
6deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ■
2deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђі
%deep_coffea_model/block2_conv1/Conv1DConv2D9deep_coffea_model/block2_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block2_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block2_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block2_conv1/BiasAddBiasAdd6deep_coffea_model/block2_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block2_conv1/ReluRelu/deep_coffea_model/block2_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block2_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block2_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block2_conv1/Relu:activations:0=deep_coffea_model/block2_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block2_conv2/Conv1DConv2D9deep_coffea_model/block2_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block2_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block2_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block2_conv2/BiasAddBiasAdd6deep_coffea_model/block2_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block2_conv2/ReluRelu/deep_coffea_model/block2_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђn
,deep_coffea_model/block2_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :█
(deep_coffea_model/block2_pool/ExpandDims
ExpandDims1deep_coffea_model/block2_conv2/Relu:activations:05deep_coffea_model/block2_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђл
%deep_coffea_model/block2_pool/MaxPoolMaxPool1deep_coffea_model/block2_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
«
%deep_coffea_model/block2_pool/SqueezeSqueeze.deep_coffea_model/block2_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
ю
)deep_coffea_model/block2_dropout/IdentityIdentity.deep_coffea_model/block2_pool/Squeeze:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block3_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        В
0deep_coffea_model/block3_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block2_dropout/Identity:output:0=deep_coffea_model/block3_conv1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block3_conv1/Conv1DConv2D9deep_coffea_model/block3_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block3_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block3_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block3_conv1/BiasAddBiasAdd6deep_coffea_model/block3_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block3_conv1/ReluRelu/deep_coffea_model/block3_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block3_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block3_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block3_conv1/Relu:activations:0=deep_coffea_model/block3_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block3_conv2/Conv1DConv2D9deep_coffea_model/block3_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block3_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block3_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block3_conv2/BiasAddBiasAdd6deep_coffea_model/block3_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block3_conv2/ReluRelu/deep_coffea_model/block3_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђn
,deep_coffea_model/block3_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :█
(deep_coffea_model/block3_pool/ExpandDims
ExpandDims1deep_coffea_model/block3_conv2/Relu:activations:05deep_coffea_model/block3_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђл
%deep_coffea_model/block3_pool/MaxPoolMaxPool1deep_coffea_model/block3_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
«
%deep_coffea_model/block3_pool/SqueezeSqueeze.deep_coffea_model/block3_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
r
!deep_coffea_model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       й
#deep_coffea_model/flatten_1/ReshapeReshape.deep_coffea_model/block3_pool/Squeeze:output:0*deep_coffea_model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђ▒
3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOpReadVariableOp<deep_coffea_model_featuresvec_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0╦
$deep_coffea_model/FeaturesVec/MatMulMatMul,deep_coffea_model/flatten_1/Reshape:output:0;deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @«
4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOpReadVariableOp=deep_coffea_model_featuresvec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
%deep_coffea_model/FeaturesVec/BiasAddBiasAdd.deep_coffea_model/FeaturesVec/MatMul:product:0<deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @}
IdentityIdentity.deep_coffea_model/FeaturesVec/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @Њ	
NoOpNoOp5^deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp4^deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp6^deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 2l
4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp2j
3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp2n
5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
┌
Ё
6__inference_deep_coffea_model_layer_call_fn_6539620261	
input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@@
	unknown_6:@ 
	unknown_7:@ђ
	unknown_8:	ђ!
	unknown_9:ђђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ"

unknown_13:ђђ

unknown_14:	ђ

unknown_15:	ђ@

unknown_16:@
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*S
ToutK
I2G*
_collective_manager_ids
 *Ч
_output_shapesж
Т:         @:	ђ@:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         ђ: :         ђ:         ђ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ:         ђ:         ђ:         @:@ђ:         @:         @:         @:         @: :         @:         @:         @:         @:         @:         @:@@:         @:         @:         @:          : @:          :          :          :          : :          :         c :         c :         c :         c :         c :  :         c :         c :         c :         c: :         c:         c:         c:         d*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539618594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d

_user_specified_nameinput
Эћ
й'
#__forward__wrapped_model_6539619961
	input_1_0`
Jdeep_coffea_model_block0_conv1_conv1d_expanddims_1_readvariableop_resource: L
>deep_coffea_model_block0_conv1_biasadd_readvariableop_resource: `
Jdeep_coffea_model_block0_conv2_conv1d_expanddims_1_readvariableop_resource:  L
>deep_coffea_model_block0_conv2_biasadd_readvariableop_resource: `
Jdeep_coffea_model_block1_conv1_conv1d_expanddims_1_readvariableop_resource: @L
>deep_coffea_model_block1_conv1_biasadd_readvariableop_resource:@`
Jdeep_coffea_model_block1_conv2_conv1d_expanddims_1_readvariableop_resource:@@L
>deep_coffea_model_block1_conv2_biasadd_readvariableop_resource:@a
Jdeep_coffea_model_block2_conv1_conv1d_expanddims_1_readvariableop_resource:@ђM
>deep_coffea_model_block2_conv1_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block2_conv2_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block2_conv2_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block3_conv1_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block3_conv1_biasadd_readvariableop_resource:	ђb
Jdeep_coffea_model_block3_conv2_conv1d_expanddims_1_readvariableop_resource:ђђM
>deep_coffea_model_block3_conv2_biasadd_readvariableop_resource:	ђO
<deep_coffea_model_featuresvec_matmul_readvariableop_resource:	ђ@K
=deep_coffea_model_featuresvec_biasadd_readvariableop_resource:@
identity7
3deep_coffea_model_featuresvec_matmul_readvariableop'
#deep_coffea_model_flatten_1_reshape)
%deep_coffea_model_block3_pool_squeeze)
%deep_coffea_model_block3_pool_maxpool,
(deep_coffea_model_block3_pool_expanddims'
#deep_coffea_model_block3_conv2_relu)
%deep_coffea_model_block3_conv2_conv1d4
0deep_coffea_model_block3_conv2_conv1d_expanddims6
2deep_coffea_model_block3_conv2_conv1d_expanddims_1'
#deep_coffea_model_block3_conv1_relu)
%deep_coffea_model_block3_conv1_conv1d4
0deep_coffea_model_block3_conv1_conv1d_expanddims6
2deep_coffea_model_block3_conv1_conv1d_expanddims_1-
)deep_coffea_model_block2_dropout_identity)
%deep_coffea_model_block2_pool_maxpool,
(deep_coffea_model_block2_pool_expanddims'
#deep_coffea_model_block2_conv2_relu)
%deep_coffea_model_block2_conv2_conv1d4
0deep_coffea_model_block2_conv2_conv1d_expanddims6
2deep_coffea_model_block2_conv2_conv1d_expanddims_1'
#deep_coffea_model_block2_conv1_relu)
%deep_coffea_model_block2_conv1_conv1d4
0deep_coffea_model_block2_conv1_conv1d_expanddims6
2deep_coffea_model_block2_conv1_conv1d_expanddims_1-
)deep_coffea_model_block1_dropout_identity)
%deep_coffea_model_block1_pool_maxpool,
(deep_coffea_model_block1_pool_expanddims'
#deep_coffea_model_block1_conv2_relu)
%deep_coffea_model_block1_conv2_conv1d4
0deep_coffea_model_block1_conv2_conv1d_expanddims6
2deep_coffea_model_block1_conv2_conv1d_expanddims_1'
#deep_coffea_model_block1_conv1_relu)
%deep_coffea_model_block1_conv1_conv1d4
0deep_coffea_model_block1_conv1_conv1d_expanddims6
2deep_coffea_model_block1_conv1_conv1d_expanddims_1-
)deep_coffea_model_block0_dropout_identity)
%deep_coffea_model_block0_pool_maxpool,
(deep_coffea_model_block0_pool_expanddims'
#deep_coffea_model_block0_conv2_relu)
%deep_coffea_model_block0_conv2_conv1d4
0deep_coffea_model_block0_conv2_conv1d_expanddims6
2deep_coffea_model_block0_conv2_conv1d_expanddims_1'
#deep_coffea_model_block0_conv1_relu)
%deep_coffea_model_block0_conv1_conv1d4
0deep_coffea_model_block0_conv1_conv1d_expanddims6
2deep_coffea_model_block0_conv1_conv1d_expanddims_1
deep_coffea_model_sub#
deep_coffea_model_strided_slice%
!deep_coffea_model_strided_slice_1
input_1ѕб4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOpб3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOpб5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpбAdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpб5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpбAdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpz
%deep_coffea_model/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           |
'deep_coffea_model/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            |
'deep_coffea_model/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         │
deep_coffea_model/strided_sliceStridedSlice	input_1_0.deep_coffea_model/strided_slice/stack:output:00deep_coffea_model/strided_slice/stack_1:output:00deep_coffea_model/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_mask|
'deep_coffea_model/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)deep_coffea_model/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            ~
)deep_coffea_model/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╗
!deep_coffea_model/strided_slice_1StridedSlice	input_1_00deep_coffea_model/strided_slice_1/stack:output:02deep_coffea_model/strided_slice_1/stack_1:output:02deep_coffea_model/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         c*

begin_mask*
end_maskе
deep_coffea_model/subSub(deep_coffea_model/strided_slice:output:0*deep_coffea_model/strided_slice_1:output:0*
T0*+
_output_shapes
:         c
4deep_coffea_model/block0_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        м
0deep_coffea_model/block0_conv1/Conv1D/ExpandDims
ExpandDimsdeep_coffea_model/sub:z:0=deep_coffea_model/block0_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cл
Adeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block0_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0x
6deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ѕ
%deep_coffea_model/block0_conv1/Conv1DConv2D9deep_coffea_model/block0_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
Й
-deep_coffea_model/block0_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block0_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ░
5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block0_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0я
&deep_coffea_model/block0_conv1/BiasAddBiasAdd6deep_coffea_model/block0_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c њ
#deep_coffea_model/block0_conv1/ReluRelu/deep_coffea_model/block0_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         c 
4deep_coffea_model/block0_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ж
0deep_coffea_model/block0_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block0_conv1/Relu:activations:0=deep_coffea_model/block0_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c л
Adeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block0_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0x
6deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ѕ
%deep_coffea_model/block0_conv2/Conv1DConv2D9deep_coffea_model/block0_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
Й
-deep_coffea_model/block0_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block0_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        ░
5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block0_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0я
&deep_coffea_model/block0_conv2/BiasAddBiasAdd6deep_coffea_model/block0_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c њ
#deep_coffea_model/block0_conv2/ReluRelu/deep_coffea_model/block0_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         c n
,deep_coffea_model/block0_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :┌
(deep_coffea_model/block0_pool/ExpandDims
ExpandDims1deep_coffea_model/block0_conv2/Relu:activations:05deep_coffea_model/block0_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         c ¤
%deep_coffea_model/block0_pool/MaxPoolMaxPool1deep_coffea_model/block0_pool/ExpandDims:output:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
Г
%deep_coffea_model/block0_pool/SqueezeSqueeze.deep_coffea_model/block0_pool/MaxPool:output:0*
T0*+
_output_shapes
:          *
squeeze_dims
Џ
)deep_coffea_model/block0_dropout/IdentityIdentity.deep_coffea_model/block0_pool/Squeeze:output:0*
T0*+
_output_shapes
:          
4deep_coffea_model/block1_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block1_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block0_dropout/Identity:output:0=deep_coffea_model/block1_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          л
Adeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block1_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0x
6deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @Ѕ
%deep_coffea_model/block1_conv1/Conv1DConv2D9deep_coffea_model/block1_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Й
-deep_coffea_model/block1_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block1_conv1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ░
5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
&deep_coffea_model/block1_conv1/BiasAddBiasAdd6deep_coffea_model/block1_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @њ
#deep_coffea_model/block1_conv1/ReluRelu/deep_coffea_model/block1_conv1/BiasAdd:output:0*
T0*+
_output_shapes
:         @
4deep_coffea_model/block1_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ж
0deep_coffea_model/block1_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block1_conv1/Relu:activations:0=deep_coffea_model/block1_conv2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @л
Adeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block1_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0x
6deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ѕ
%deep_coffea_model/block1_conv2/Conv1DConv2D9deep_coffea_model/block1_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Й
-deep_coffea_model/block1_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block1_conv2/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ░
5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
&deep_coffea_model/block1_conv2/BiasAddBiasAdd6deep_coffea_model/block1_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @њ
#deep_coffea_model/block1_conv2/ReluRelu/deep_coffea_model/block1_conv2/BiasAdd:output:0*
T0*+
_output_shapes
:         @n
,deep_coffea_model/block1_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :┌
(deep_coffea_model/block1_pool/ExpandDims
ExpandDims1deep_coffea_model/block1_conv2/Relu:activations:05deep_coffea_model/block1_pool/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @¤
%deep_coffea_model/block1_pool/MaxPoolMaxPool1deep_coffea_model/block1_pool/ExpandDims:output:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
Г
%deep_coffea_model/block1_pool/SqueezeSqueeze.deep_coffea_model/block1_pool/MaxPool:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims
Џ
)deep_coffea_model/block1_dropout/IdentityIdentity.deep_coffea_model/block1_pool/Squeeze:output:0*
T0*+
_output_shapes
:         @
4deep_coffea_model/block2_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block2_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block1_dropout/Identity:output:0=deep_coffea_model/block2_conv1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Л
Adeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block2_conv1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0x
6deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ■
2deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђі
%deep_coffea_model/block2_conv1/Conv1DConv2D9deep_coffea_model/block2_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block2_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block2_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block2_conv1/BiasAddBiasAdd6deep_coffea_model/block2_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block2_conv1/ReluRelu/deep_coffea_model/block2_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block2_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block2_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block2_conv1/Relu:activations:0=deep_coffea_model/block2_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block2_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block2_conv2/Conv1DConv2D9deep_coffea_model/block2_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block2_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block2_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block2_conv2/BiasAddBiasAdd6deep_coffea_model/block2_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block2_conv2/ReluRelu/deep_coffea_model/block2_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђn
,deep_coffea_model/block2_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :█
(deep_coffea_model/block2_pool/ExpandDims
ExpandDims1deep_coffea_model/block2_conv2/Relu:activations:05deep_coffea_model/block2_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђл
%deep_coffea_model/block2_pool/MaxPoolMaxPool1deep_coffea_model/block2_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
«
%deep_coffea_model/block2_pool/SqueezeSqueeze.deep_coffea_model/block2_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
ю
)deep_coffea_model/block2_dropout/IdentityIdentity.deep_coffea_model/block2_pool/Squeeze:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block3_conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        В
0deep_coffea_model/block3_conv1/Conv1D/ExpandDims
ExpandDims2deep_coffea_model/block2_dropout/Identity:output:0=deep_coffea_model/block3_conv1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block3_conv1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block3_conv1/Conv1DConv2D9deep_coffea_model/block3_conv1/Conv1D/ExpandDims:output:0;deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block3_conv1/Conv1D/SqueezeSqueeze.deep_coffea_model/block3_conv1/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block3_conv1/BiasAddBiasAdd6deep_coffea_model/block3_conv1/Conv1D/Squeeze:output:0=deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block3_conv1/ReluRelu/deep_coffea_model/block3_conv1/BiasAdd:output:0*
T0*,
_output_shapes
:         ђ
4deep_coffea_model/block3_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        в
0deep_coffea_model/block3_conv2/Conv1D/ExpandDims
ExpandDims1deep_coffea_model/block3_conv1/Relu:activations:0=deep_coffea_model/block3_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђм
Adeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJdeep_coffea_model_block3_conv2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0x
6deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
2deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1
ExpandDimsIdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0?deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђі
%deep_coffea_model/block3_conv2/Conv1DConv2D9deep_coffea_model/block3_conv2/Conv1D/ExpandDims:output:0;deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
┐
-deep_coffea_model/block3_conv2/Conv1D/SqueezeSqueeze.deep_coffea_model/block3_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        ▒
5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp>deep_coffea_model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
&deep_coffea_model/block3_conv2/BiasAddBiasAdd6deep_coffea_model/block3_conv2/Conv1D/Squeeze:output:0=deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЊ
#deep_coffea_model/block3_conv2/ReluRelu/deep_coffea_model/block3_conv2/BiasAdd:output:0*
T0*,
_output_shapes
:         ђn
,deep_coffea_model/block3_pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :█
(deep_coffea_model/block3_pool/ExpandDims
ExpandDims1deep_coffea_model/block3_conv2/Relu:activations:05deep_coffea_model/block3_pool/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђл
%deep_coffea_model/block3_pool/MaxPoolMaxPool1deep_coffea_model/block3_pool/ExpandDims:output:0*0
_output_shapes
:         ђ*
ksize
*
paddingSAME*
strides
«
%deep_coffea_model/block3_pool/SqueezeSqueeze.deep_coffea_model/block3_pool/MaxPool:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims
r
!deep_coffea_model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       й
#deep_coffea_model/flatten_1/ReshapeReshape.deep_coffea_model/block3_pool/Squeeze:output:0*deep_coffea_model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђ▒
3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOpReadVariableOp<deep_coffea_model_featuresvec_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0╦
$deep_coffea_model/FeaturesVec/MatMulMatMul,deep_coffea_model/flatten_1/Reshape:output:0;deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @«
4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOpReadVariableOp=deep_coffea_model_featuresvec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
%deep_coffea_model/FeaturesVec/BiasAddBiasAdd.deep_coffea_model/FeaturesVec/MatMul:product:0<deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @}
IdentityIdentity.deep_coffea_model/FeaturesVec/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @Њ	
NoOpNoOp5^deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp4^deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp6^deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOpB^deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp6^deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOpB^deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%deep_coffea_model_block0_conv1_conv1d.deep_coffea_model/block0_conv1/Conv1D:output:0"m
0deep_coffea_model_block0_conv1_conv1d_expanddims9deep_coffea_model/block0_conv1/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block0_conv1_conv1d_expanddims_1;deep_coffea_model/block0_conv1/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block0_conv1_relu1deep_coffea_model/block0_conv1/Relu:activations:0"W
%deep_coffea_model_block0_conv2_conv1d.deep_coffea_model/block0_conv2/Conv1D:output:0"m
0deep_coffea_model_block0_conv2_conv1d_expanddims9deep_coffea_model/block0_conv2/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block0_conv2_conv1d_expanddims_1;deep_coffea_model/block0_conv2/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block0_conv2_relu1deep_coffea_model/block0_conv2/Relu:activations:0"_
)deep_coffea_model_block0_dropout_identity2deep_coffea_model/block0_dropout/Identity:output:0"]
(deep_coffea_model_block0_pool_expanddims1deep_coffea_model/block0_pool/ExpandDims:output:0"W
%deep_coffea_model_block0_pool_maxpool.deep_coffea_model/block0_pool/MaxPool:output:0"W
%deep_coffea_model_block1_conv1_conv1d.deep_coffea_model/block1_conv1/Conv1D:output:0"m
0deep_coffea_model_block1_conv1_conv1d_expanddims9deep_coffea_model/block1_conv1/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block1_conv1_conv1d_expanddims_1;deep_coffea_model/block1_conv1/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block1_conv1_relu1deep_coffea_model/block1_conv1/Relu:activations:0"W
%deep_coffea_model_block1_conv2_conv1d.deep_coffea_model/block1_conv2/Conv1D:output:0"m
0deep_coffea_model_block1_conv2_conv1d_expanddims9deep_coffea_model/block1_conv2/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block1_conv2_conv1d_expanddims_1;deep_coffea_model/block1_conv2/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block1_conv2_relu1deep_coffea_model/block1_conv2/Relu:activations:0"_
)deep_coffea_model_block1_dropout_identity2deep_coffea_model/block1_dropout/Identity:output:0"]
(deep_coffea_model_block1_pool_expanddims1deep_coffea_model/block1_pool/ExpandDims:output:0"W
%deep_coffea_model_block1_pool_maxpool.deep_coffea_model/block1_pool/MaxPool:output:0"W
%deep_coffea_model_block2_conv1_conv1d.deep_coffea_model/block2_conv1/Conv1D:output:0"m
0deep_coffea_model_block2_conv1_conv1d_expanddims9deep_coffea_model/block2_conv1/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block2_conv1_conv1d_expanddims_1;deep_coffea_model/block2_conv1/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block2_conv1_relu1deep_coffea_model/block2_conv1/Relu:activations:0"W
%deep_coffea_model_block2_conv2_conv1d.deep_coffea_model/block2_conv2/Conv1D:output:0"m
0deep_coffea_model_block2_conv2_conv1d_expanddims9deep_coffea_model/block2_conv2/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block2_conv2_conv1d_expanddims_1;deep_coffea_model/block2_conv2/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block2_conv2_relu1deep_coffea_model/block2_conv2/Relu:activations:0"_
)deep_coffea_model_block2_dropout_identity2deep_coffea_model/block2_dropout/Identity:output:0"]
(deep_coffea_model_block2_pool_expanddims1deep_coffea_model/block2_pool/ExpandDims:output:0"W
%deep_coffea_model_block2_pool_maxpool.deep_coffea_model/block2_pool/MaxPool:output:0"W
%deep_coffea_model_block3_conv1_conv1d.deep_coffea_model/block3_conv1/Conv1D:output:0"m
0deep_coffea_model_block3_conv1_conv1d_expanddims9deep_coffea_model/block3_conv1/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block3_conv1_conv1d_expanddims_1;deep_coffea_model/block3_conv1/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block3_conv1_relu1deep_coffea_model/block3_conv1/Relu:activations:0"W
%deep_coffea_model_block3_conv2_conv1d.deep_coffea_model/block3_conv2/Conv1D:output:0"m
0deep_coffea_model_block3_conv2_conv1d_expanddims9deep_coffea_model/block3_conv2/Conv1D/ExpandDims:output:0"q
2deep_coffea_model_block3_conv2_conv1d_expanddims_1;deep_coffea_model/block3_conv2/Conv1D/ExpandDims_1:output:0"X
#deep_coffea_model_block3_conv2_relu1deep_coffea_model/block3_conv2/Relu:activations:0"]
(deep_coffea_model_block3_pool_expanddims1deep_coffea_model/block3_pool/ExpandDims:output:0"W
%deep_coffea_model_block3_pool_maxpool.deep_coffea_model/block3_pool/MaxPool:output:0"W
%deep_coffea_model_block3_pool_squeeze.deep_coffea_model/block3_pool/Squeeze:output:0"r
3deep_coffea_model_featuresvec_matmul_readvariableop;deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp:value:0"S
#deep_coffea_model_flatten_1_reshape,deep_coffea_model/flatten_1/Reshape:output:0"K
deep_coffea_model_strided_slice(deep_coffea_model/strided_slice:output:0"O
!deep_coffea_model_strided_slice_1*deep_coffea_model/strided_slice_1:output:0"2
deep_coffea_model_subdeep_coffea_model/sub:z:0"
identityIdentity:output:0"
input_1	input_1_0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : *W
backward_function_name=;__inference___backward__wrapped_model_6539619617_65396199622l
4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp4deep_coffea_model/FeaturesVec/BiasAdd/ReadVariableOp2j
3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp3deep_coffea_model/FeaturesVec/MatMul/ReadVariableOp2n
5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block0_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block0_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block0_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block0_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block1_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block1_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block1_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block1_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block2_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block2_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block2_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block2_conv2/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp5deep_coffea_model/block3_conv1/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block3_conv1/Conv1D/ExpandDims_1/ReadVariableOp2n
5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp5deep_coffea_model/block3_conv2/BiasAdd/ReadVariableOp2є
Adeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOpAdeep_coffea_model/block3_conv2/Conv1D/ExpandDims_1/ReadVariableOp:T P
+
_output_shapes
:         d
!
_user_specified_name	input_1
Ш
»
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539617166
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?j
dropout/MulMulinputs_0dropout/Const:output:0*
T0*+
_output_shapes
:         @E
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         @*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         @s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         @m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         @]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         @" 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @*ђ
backward_function_namefd__inference___backward_block1_dropout_layer_call_and_return_conditional_losses_6539617127_6539617167:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
¤
Џ
L__inference_block1_conv1_layer_call_and_return_conditional_losses_6539620676

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
С
­
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539613117
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block3_conv1_layer_call_and_return_conditional_losses_6539613078_653961311820
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
к
O
3__inference_block1_dropout_layer_call_fn_6539620910

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539611724d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
┼
e
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539620565

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
у
б
1__inference_block0_conv2_layer_call_fn_6539620630

inputs
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c :  :         c *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539611308s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         c `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         c 
 
_user_specified_nameinputs
¤
Џ
L__inference_block0_conv1_layer_call_and_return_conditional_losses_6539620616

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         cњ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         c *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         c *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         c e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         c ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         c: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         c
 
_user_specified_nameinputs
ы
l
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539611724

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         @_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ѕ
L
0__inference_block1_pool_layer_call_fn_6539620844

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *T
fORM
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539611019v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
к
O
3__inference_block0_dropout_layer_call_fn_6539620883

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539611389d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
н
ь
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539614121
inputs_0A
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : *~
backward_function_namedb__inference___backward_block1_conv2_layer_call_and_return_conditional_losses_6539614082_653961412220
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
¤
g
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539620878

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
»Ќ
в
O__forward_deep_coffea_model_layer_call_and_return_conditional_losses_6539615827
input_0-
block0_conv1_6539611184: %
block0_conv1_6539611186: -
block0_conv2_6539611310:  %
block0_conv2_6539611312: -
block1_conv1_6539611519: @%
block1_conv1_6539611521:@-
block1_conv2_6539611645:@@%
block1_conv2_6539611647:@.
block2_conv1_6539611854:@ђ&
block2_conv1_6539611856:	ђ/
block2_conv2_6539611980:ђђ&
block2_conv2_6539611982:	ђ/
block3_conv1_6539612189:ђђ&
block3_conv1_6539612191:	ђ/
block3_conv2_6539612315:ђђ&
block3_conv2_6539612317:	ђ)
featuresvec_6539612487:	ђ@$
featuresvec_6539612489:@
identity'
#featuresvec_statefulpartitionedcall)
%featuresvec_statefulpartitionedcall_0
flatten_1_partitionedcall
block3_pool_partitionedcall!
block3_pool_partitionedcall_0!
block3_pool_partitionedcall_1(
$block3_conv2_statefulpartitionedcall*
&block3_conv2_statefulpartitionedcall_0*
&block3_conv2_statefulpartitionedcall_1*
&block3_conv2_statefulpartitionedcall_2*
&block3_conv2_statefulpartitionedcall_3(
$block3_conv1_statefulpartitionedcall*
&block3_conv1_statefulpartitionedcall_0*
&block3_conv1_statefulpartitionedcall_1*
&block3_conv1_statefulpartitionedcall_2*
&block3_conv1_statefulpartitionedcall_3
block2_pool_partitionedcall!
block2_pool_partitionedcall_0!
block2_pool_partitionedcall_1(
$block2_conv2_statefulpartitionedcall*
&block2_conv2_statefulpartitionedcall_0*
&block2_conv2_statefulpartitionedcall_1*
&block2_conv2_statefulpartitionedcall_2*
&block2_conv2_statefulpartitionedcall_3(
$block2_conv1_statefulpartitionedcall*
&block2_conv1_statefulpartitionedcall_0*
&block2_conv1_statefulpartitionedcall_1*
&block2_conv1_statefulpartitionedcall_2*
&block2_conv1_statefulpartitionedcall_3
block1_pool_partitionedcall!
block1_pool_partitionedcall_0!
block1_pool_partitionedcall_1(
$block1_conv2_statefulpartitionedcall*
&block1_conv2_statefulpartitionedcall_0*
&block1_conv2_statefulpartitionedcall_1*
&block1_conv2_statefulpartitionedcall_2*
&block1_conv2_statefulpartitionedcall_3(
$block1_conv1_statefulpartitionedcall*
&block1_conv1_statefulpartitionedcall_0*
&block1_conv1_statefulpartitionedcall_1*
&block1_conv1_statefulpartitionedcall_2*
&block1_conv1_statefulpartitionedcall_3
block0_pool_partitionedcall!
block0_pool_partitionedcall_0!
block0_pool_partitionedcall_1(
$block0_conv2_statefulpartitionedcall*
&block0_conv2_statefulpartitionedcall_0*
&block0_conv2_statefulpartitionedcall_1*
&block0_conv2_statefulpartitionedcall_2*
&block0_conv2_statefulpartitionedcall_3(
$block0_conv1_statefulpartitionedcall*
&block0_conv1_statefulpartitionedcall_0*
&block0_conv1_statefulpartitionedcall_1*
&block0_conv1_statefulpartitionedcall_2*
&block0_conv1_statefulpartitionedcall_3
strided_slice
strided_slice_1	
inputѕб#FeaturesVec/StatefulPartitionedCallб$block0_conv1/StatefulPartitionedCallб$block0_conv2/StatefulPartitionedCallб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Й
strided_slice_0StridedSliceinput_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         к
strided_slice_1_0StridedSliceinput_0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*

begin_mask*
end_maskv
subSubstrided_slice_0:output:0strided_slice_1_0:output:0*
T0*+
_output_shapes
:         cњ
$block0_conv1/StatefulPartitionedCallStatefulPartitionedCallsub:z:0block0_conv1_6539611184block0_conv1_6539611186*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c: :         c*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv1_layer_call_and_return_conditional_losses_6539614995И
$block0_conv2/StatefulPartitionedCallStatefulPartitionedCall-block0_conv1/StatefulPartitionedCall:output:0block0_conv2_6539611310block0_conv2_6539611312*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         c :         c :         c :         c :  :         c *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block0_conv2_layer_call_and_return_conditional_losses_6539614747└
block0_pool/PartitionedCallPartitionedCall-block0_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:          :          :         c :         c * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block0_pool_layer_call_and_return_conditional_losses_6539614616ь
block0_dropout/PartitionedCallPartitionedCall$block0_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block0_dropout_layer_call_and_return_conditional_losses_6539614573▓
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall'block0_dropout/PartitionedCall:output:0block1_conv1_6539611519block1_conv1_6539611521*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:          : @:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv1_layer_call_and_return_conditional_losses_6539614369И
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_6539611645block1_conv2_6539611647*
Tin
2*
Tout

2*
_collective_manager_ids
 *Б
_output_shapesљ
Ї:         @:         @:         @:         @:@@:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539614121└
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *x
_output_shapesf
d:         @:         @:         @:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block1_pool_layer_call_and_return_conditional_losses_6539613990ь
block1_dropout/PartitionedCallPartitionedCall$block1_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block1_dropout_layer_call_and_return_conditional_losses_6539613947Х
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall'block1_dropout/PartitionedCall:output:0block2_conv1_6539611854block2_conv1_6539611856*
Tin
2*
Tout

2*
_collective_manager_ids
 *Д
_output_shapesћ
Љ:         ђ:         ђ:         ђ:         @:@ђ:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv1_layer_call_and_return_conditional_losses_6539613743┐
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_6539611980block2_conv2_6539611982*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539613495─
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539613364Ь
block2_dropout/PartitionedCallPartitionedCall$block2_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539613321╣
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall'block2_dropout/PartitionedCall:output:0block3_conv1_6539612189block3_conv1_6539612191*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv1_layer_call_and_return_conditional_losses_6539613117┐
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_6539612315block3_conv2_6539612317*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block3_conv2_layer_call_and_return_conditional_losses_6539612869─
block3_pool/PartitionedCallPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *|
_output_shapesj
h:         ђ:         ђ:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_block3_pool_layer_call_and_return_conditional_losses_6539612738щ
flatten_1/PartitionedCallPartitionedCall$block3_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:         ђ:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612673╚
#FeaturesVec/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0featuresvec_6539612487featuresvec_6539612489*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         @:	ђ@:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__forward_FeaturesVec_layer_call_and_return_conditional_losses_6539612522{
IdentityIdentity,FeaturesVec/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ц
NoOpNoOp$^FeaturesVec/StatefulPartitionedCall%^block0_conv1/StatefulPartitionedCall%^block0_conv2/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "U
$block0_conv1_statefulpartitionedcall-block0_conv1/StatefulPartitionedCall:output:1"W
&block0_conv1_statefulpartitionedcall_0-block0_conv1/StatefulPartitionedCall:output:2"W
&block0_conv1_statefulpartitionedcall_1-block0_conv1/StatefulPartitionedCall:output:3"W
&block0_conv1_statefulpartitionedcall_2-block0_conv1/StatefulPartitionedCall:output:4"W
&block0_conv1_statefulpartitionedcall_3-block0_conv1/StatefulPartitionedCall:output:5"U
$block0_conv2_statefulpartitionedcall-block0_conv2/StatefulPartitionedCall:output:1"W
&block0_conv2_statefulpartitionedcall_0-block0_conv2/StatefulPartitionedCall:output:2"W
&block0_conv2_statefulpartitionedcall_1-block0_conv2/StatefulPartitionedCall:output:3"W
&block0_conv2_statefulpartitionedcall_2-block0_conv2/StatefulPartitionedCall:output:4"W
&block0_conv2_statefulpartitionedcall_3-block0_conv2/StatefulPartitionedCall:output:5"C
block0_pool_partitionedcall$block0_pool/PartitionedCall:output:1"E
block0_pool_partitionedcall_0$block0_pool/PartitionedCall:output:2"E
block0_pool_partitionedcall_1$block0_pool/PartitionedCall:output:3"U
$block1_conv1_statefulpartitionedcall-block1_conv1/StatefulPartitionedCall:output:1"W
&block1_conv1_statefulpartitionedcall_0-block1_conv1/StatefulPartitionedCall:output:2"W
&block1_conv1_statefulpartitionedcall_1-block1_conv1/StatefulPartitionedCall:output:3"W
&block1_conv1_statefulpartitionedcall_2-block1_conv1/StatefulPartitionedCall:output:4"W
&block1_conv1_statefulpartitionedcall_3-block1_conv1/StatefulPartitionedCall:output:5"U
$block1_conv2_statefulpartitionedcall-block1_conv2/StatefulPartitionedCall:output:1"W
&block1_conv2_statefulpartitionedcall_0-block1_conv2/StatefulPartitionedCall:output:2"W
&block1_conv2_statefulpartitionedcall_1-block1_conv2/StatefulPartitionedCall:output:3"W
&block1_conv2_statefulpartitionedcall_2-block1_conv2/StatefulPartitionedCall:output:4"W
&block1_conv2_statefulpartitionedcall_3-block1_conv2/StatefulPartitionedCall:output:5"C
block1_pool_partitionedcall$block1_pool/PartitionedCall:output:1"E
block1_pool_partitionedcall_0$block1_pool/PartitionedCall:output:2"E
block1_pool_partitionedcall_1$block1_pool/PartitionedCall:output:3"U
$block2_conv1_statefulpartitionedcall-block2_conv1/StatefulPartitionedCall:output:1"W
&block2_conv1_statefulpartitionedcall_0-block2_conv1/StatefulPartitionedCall:output:2"W
&block2_conv1_statefulpartitionedcall_1-block2_conv1/StatefulPartitionedCall:output:3"W
&block2_conv1_statefulpartitionedcall_2-block2_conv1/StatefulPartitionedCall:output:4"W
&block2_conv1_statefulpartitionedcall_3-block2_conv1/StatefulPartitionedCall:output:5"U
$block2_conv2_statefulpartitionedcall-block2_conv2/StatefulPartitionedCall:output:1"W
&block2_conv2_statefulpartitionedcall_0-block2_conv2/StatefulPartitionedCall:output:2"W
&block2_conv2_statefulpartitionedcall_1-block2_conv2/StatefulPartitionedCall:output:3"W
&block2_conv2_statefulpartitionedcall_2-block2_conv2/StatefulPartitionedCall:output:4"W
&block2_conv2_statefulpartitionedcall_3-block2_conv2/StatefulPartitionedCall:output:5"C
block2_pool_partitionedcall$block2_pool/PartitionedCall:output:1"E
block2_pool_partitionedcall_0$block2_pool/PartitionedCall:output:2"E
block2_pool_partitionedcall_1$block2_pool/PartitionedCall:output:3"U
$block3_conv1_statefulpartitionedcall-block3_conv1/StatefulPartitionedCall:output:1"W
&block3_conv1_statefulpartitionedcall_0-block3_conv1/StatefulPartitionedCall:output:2"W
&block3_conv1_statefulpartitionedcall_1-block3_conv1/StatefulPartitionedCall:output:3"W
&block3_conv1_statefulpartitionedcall_2-block3_conv1/StatefulPartitionedCall:output:4"W
&block3_conv1_statefulpartitionedcall_3-block3_conv1/StatefulPartitionedCall:output:5"U
$block3_conv2_statefulpartitionedcall-block3_conv2/StatefulPartitionedCall:output:1"W
&block3_conv2_statefulpartitionedcall_0-block3_conv2/StatefulPartitionedCall:output:2"W
&block3_conv2_statefulpartitionedcall_1-block3_conv2/StatefulPartitionedCall:output:3"W
&block3_conv2_statefulpartitionedcall_2-block3_conv2/StatefulPartitionedCall:output:4"W
&block3_conv2_statefulpartitionedcall_3-block3_conv2/StatefulPartitionedCall:output:5"C
block3_pool_partitionedcall$block3_pool/PartitionedCall:output:1"E
block3_pool_partitionedcall_0$block3_pool/PartitionedCall:output:2"E
block3_pool_partitionedcall_1$block3_pool/PartitionedCall:output:3"S
#featuresvec_statefulpartitionedcall,FeaturesVec/StatefulPartitionedCall:output:1"U
%featuresvec_statefulpartitionedcall_0,FeaturesVec/StatefulPartitionedCall:output:2"?
flatten_1_partitionedcall"flatten_1/PartitionedCall:output:1"
identityIdentity:output:0"
inputinput_0")
strided_slicestrided_slice_0:output:0"-
strided_slice_1strided_slice_1_0:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         d: : : : : : : : : : : : : : : : : : *Ѓ
backward_function_nameig__inference___backward_deep_coffea_model_layer_call_and_return_conditional_losses_6539615524_65396158282J
#FeaturesVec/StatefulPartitionedCall#FeaturesVec/StatefulPartitionedCall2L
$block0_conv1/StatefulPartitionedCall$block0_conv1/StatefulPartitionedCall2L
$block0_conv2/StatefulPartitionedCall$block0_conv2/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall:R N
+
_output_shapes
:         d

_user_specified_nameinput
З
Ц
1__inference_block2_conv2_layer_call_fn_6539620750

inputs
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout

2*
_collective_manager_ids
 *ф
_output_shapesЌ
ћ:         ђ:         ђ:         ђ:         ђ:ђђ:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539611978t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Г

m
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620905

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:          *
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:          s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:          m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:          ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
▀
ъ
L__inference_block3_conv2_layer_call_and_return_conditional_losses_6539620826

inputsC
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ѓ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
С
q
G__forward_flatten_1_layer_call_and_return_conditional_losses_6539612427
inputs_0
identity

inputsV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       _
ReshapeReshapeinputs_0Const:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*{
backward_function_namea___inference___backward_flatten_1_layer_call_and_return_conditional_losses_6539612417_6539612428:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
■
»
L__forward_block2_dropout_layer_call_and_return_conditional_losses_6539616811
inputs_0
identity
dropout_mul
dropout_cast

inputs
dropout_constѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?k
dropout/MulMulinputs_0dropout/Const:output:0*
T0*,
_output_shapes
:         ђE
dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:б
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ђ*
dtype0*
seed2    [
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ђt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ђn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ђ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ђ" 
dropout_castdropout/Cast:y:0"'
dropout_constdropout/Const:output:0"
dropout_muldropout/Mul:z:0"
identityIdentity:output:0"
inputsinputs_0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ*ђ
backward_function_namefd__inference___backward_block2_dropout_layer_call_and_return_conditional_losses_6539616772_6539616812:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
Џ
L__inference_block1_conv2_layer_call_and_return_conditional_losses_6539620706

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
н
ь
J__forward_block1_conv2_layer_call_and_return_conditional_losses_6539611643
inputs_0A
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ѓ
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         @e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : *~
backward_function_namedb__inference___backward_block1_conv2_layer_call_and_return_conditional_losses_6539611607_653961164420
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╬	
љ
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539613364
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block2_pool_layer_call_and_return_conditional_losses_6539613342_6539613365:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ѕ
L
0__inference_block2_pool_layer_call_fn_6539620857

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *T
fORM
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539611034v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╬	
љ
I__forward_block2_pool_layer_call_and_return_conditional_losses_6539612048
inputs_0
identity
maxpool

expanddims

inputsP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "!

expanddimsExpandDims:output:0"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           *}
backward_function_nameca__inference___backward_block2_pool_layer_call_and_return_conditional_losses_6539612027_6539612049:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
С
­
J__forward_block2_conv2_layer_call_and_return_conditional_losses_6539613495
inputs_0C
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identity
relu

conv1d
conv1d_expanddims
conv1d_expanddims_1

inputsѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ё
Conv1D/ExpandDims
ExpandDimsinputs_0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ђћ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : б
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:         ђf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
conv1dConv1D:output:0"/
conv1d_expanddimsConv1D/ExpandDims:output:0"3
conv1d_expanddims_1Conv1D/ExpandDims_1:output:0"
identityIdentity:output:0"
inputsinputs_0"
reluRelu:activations:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : *~
backward_function_namedb__inference___backward_block2_conv2_layer_call_and_return_conditional_losses_6539613456_653961349620
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_defaultЏ
?
input_14
serving_default_input_1:0         d<
output_10
StatefulPartitionedCall:0         @tensorflow/serving/predict:щЁ
а
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
conv_layers
	pool_layers

dropout_layers
flatten
	dense

signatures"
_tf_keras_model
д
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
д
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ї
%trace_0
&trace_1
'trace_2
(trace_32А
6__inference_deep_coffea_model_layer_call_fn_6539615925
6__inference_deep_coffea_model_layer_call_fn_6539620150
6__inference_deep_coffea_model_layer_call_fn_6539620261
6__inference_deep_coffea_model_layer_call_fn_6539618815Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 z%trace_0z&trace_1z'trace_2z(trace_3
Э
)trace_0
*trace_1
+trace_2
,trace_32Ї
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620397
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620554
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539618936
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539619069Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 z)trace_0z*trace_1z+trace_2z,trace_3
лB═
%__inference__wrapped_model_6539610992input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Y
-0
.1
/2
03
14
25
36
47"
trackable_tuple_wrapper
=
50
61
72
83"
trackable_tuple_wrapper
6
90
:1
;2"
trackable_tuple_wrapper
Ц
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
,
Hserving_default"
signature_map
;:9 2%deep_coffea_model/block0_conv1/kernel
1:/ 2#deep_coffea_model/block0_conv1/bias
;:9  2%deep_coffea_model/block0_conv2/kernel
1:/ 2#deep_coffea_model/block0_conv2/bias
;:9 @2%deep_coffea_model/block1_conv1/kernel
1:/@2#deep_coffea_model/block1_conv1/bias
;:9@@2%deep_coffea_model/block1_conv2/kernel
1:/@2#deep_coffea_model/block1_conv2/bias
<::@ђ2%deep_coffea_model/block2_conv1/kernel
2:0ђ2#deep_coffea_model/block2_conv1/bias
=:;ђђ2%deep_coffea_model/block2_conv2/kernel
2:0ђ2#deep_coffea_model/block2_conv2/bias
=:;ђђ2%deep_coffea_model/block3_conv1/kernel
2:0ђ2#deep_coffea_model/block3_conv1/bias
=:;ђђ2%deep_coffea_model/block3_conv2/kernel
2:0ђ2#deep_coffea_model/block3_conv2/bias
7:5	ђ@2$deep_coffea_model/FeaturesVec/kernel
0:.@2"deep_coffea_model/FeaturesVec/bias
 "
trackable_list_wrapper
ъ
90
:1
;2
-3
.4
/5
06
17
28
39
410
511
612
713
814
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЄBё
6__inference_deep_coffea_model_layer_call_fn_6539615925input_1"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЁBѓ
6__inference_deep_coffea_model_layer_call_fn_6539620150input"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЁBѓ
6__inference_deep_coffea_model_layer_call_fn_6539620261input"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЄBё
6__inference_deep_coffea_model_layer_call_fn_6539618815input_1"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
аBЮ
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620397input"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
аBЮ
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620554input"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
бBЪ
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539618936input_1"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
бBЪ
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539619069input_1"Й
х▓▒
FullArgSpec
argsџ
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
П
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
bias
 O_jit_compiled_convolution_op"
_tf_keras_layer
П
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias
 V_jit_compiled_convolution_op"
_tf_keras_layer
П
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias
 ]_jit_compiled_convolution_op"
_tf_keras_layer
П
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op"
_tf_keras_layer
П
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kernel
bias
 k_jit_compiled_convolution_op"
_tf_keras_layer
П
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

kernel
bias
 r_jit_compiled_convolution_op"
_tf_keras_layer
П
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

kernel
bias
 y_jit_compiled_convolution_op"
_tf_keras_layer
я
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias
!ђ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
ё	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Њ	variables
ћtrainable_variables
Ћregularization_losses
ќ	keras_api
Ќ__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ъ_random_generator"
_tf_keras_layer
├
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses
д_random_generator"
_tf_keras_layer
├
Д	variables
еtrainable_variables
Еregularization_losses
ф	keras_api
Ф__call__
+г&call_and_return_all_conditional_losses
Г_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
З
│trace_02Н
.__inference_flatten_1_layer_call_fn_6539620559б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z│trace_0
Ј
┤trace_02­
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539620565б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ш
║trace_02О
0__inference_FeaturesVec_layer_call_fn_6539620576б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z║trace_0
Љ
╗trace_02Ы
K__inference_FeaturesVec_layer_call_and_return_conditional_losses_6539620586б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗trace_0
¤B╠
(__inference_signature_wrapper_6539620051input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
э
┴trace_02п
1__inference_block0_conv1_layer_call_fn_6539620600б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┴trace_0
њ
┬trace_02з
L__inference_block0_conv1_layer_call_and_return_conditional_losses_6539620616б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
э
╚trace_02п
1__inference_block0_conv2_layer_call_fn_6539620630б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╚trace_0
њ
╔trace_02з
L__inference_block0_conv2_layer_call_and_return_conditional_losses_6539620646б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╔trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
э
¤trace_02п
1__inference_block1_conv1_layer_call_fn_6539620660б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z¤trace_0
њ
лtrace_02з
L__inference_block1_conv1_layer_call_and_return_conditional_losses_6539620676б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zлtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
э
оtrace_02п
1__inference_block1_conv2_layer_call_fn_6539620690б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zоtrace_0
њ
Оtrace_02з
L__inference_block1_conv2_layer_call_and_return_conditional_losses_6539620706б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zОtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
э
Пtrace_02п
1__inference_block2_conv1_layer_call_fn_6539620720б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zПtrace_0
њ
яtrace_02з
L__inference_block2_conv1_layer_call_and_return_conditional_losses_6539620736б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
э
Сtrace_02п
1__inference_block2_conv2_layer_call_fn_6539620750б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0
њ
тtrace_02з
L__inference_block2_conv2_layer_call_and_return_conditional_losses_6539620766б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zтtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
э
вtrace_02п
1__inference_block3_conv1_layer_call_fn_6539620780б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zвtrace_0
њ
Вtrace_02з
L__inference_block3_conv1_layer_call_and_return_conditional_losses_6539620796б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zВtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
Ыtrace_02п
1__inference_block3_conv2_layer_call_fn_6539620810б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЫtrace_0
њ
зtrace_02з
L__inference_block3_conv2_layer_call_and_return_conditional_losses_6539620826б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zзtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
Ш
щtrace_02О
0__inference_block0_pool_layer_call_fn_6539620831б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zщtrace_0
Љ
Щtrace_02Ы
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539620839б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЩtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
Ш
ђtrace_02О
0__inference_block1_pool_layer_call_fn_6539620844б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zђtrace_0
Љ
Ђtrace_02Ы
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539620852б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
Ш
Єtrace_02О
0__inference_block2_pool_layer_call_fn_6539620857б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЄtrace_0
Љ
ѕtrace_02Ы
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539620865б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Њ	variables
ћtrainable_variables
Ћregularization_losses
Ќ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
Ш
јtrace_02О
0__inference_block3_pool_layer_call_fn_6539620870б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zјtrace_0
Љ
Јtrace_02Ы
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539620878б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
█
Ћtrace_0
ќtrace_12а
3__inference_block0_dropout_layer_call_fn_6539620883
3__inference_block0_dropout_layer_call_fn_6539620888│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЋtrace_0zќtrace_1
Љ
Ќtrace_0
ўtrace_12о
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620893
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620905│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0zўtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
█
ъtrace_0
Ъtrace_12а
3__inference_block1_dropout_layer_call_fn_6539620910
3__inference_block1_dropout_layer_call_fn_6539620915│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zъtrace_0zЪtrace_1
Љ
аtrace_0
Аtrace_12о
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620920
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620932│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zаtrace_0zАtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
Д	variables
еtrainable_variables
Еregularization_losses
Ф__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
█
Дtrace_0
еtrace_12а
3__inference_block2_dropout_layer_call_fn_6539620937
3__inference_block2_dropout_layer_call_fn_6539620942│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zДtrace_0zеtrace_1
Љ
Еtrace_0
фtrace_12о
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620947
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620959│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЕtrace_0zфtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
РB▀
.__inference_flatten_1_layer_call_fn_6539620559inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539620565inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_FeaturesVec_layer_call_fn_6539620576inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_FeaturesVec_layer_call_and_return_conditional_losses_6539620586inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block0_conv1_layer_call_fn_6539620600inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block0_conv1_layer_call_and_return_conditional_losses_6539620616inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block0_conv2_layer_call_fn_6539620630inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block0_conv2_layer_call_and_return_conditional_losses_6539620646inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block1_conv1_layer_call_fn_6539620660inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block1_conv1_layer_call_and_return_conditional_losses_6539620676inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block1_conv2_layer_call_fn_6539620690inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block1_conv2_layer_call_and_return_conditional_losses_6539620706inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block2_conv1_layer_call_fn_6539620720inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block2_conv1_layer_call_and_return_conditional_losses_6539620736inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block2_conv2_layer_call_fn_6539620750inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block2_conv2_layer_call_and_return_conditional_losses_6539620766inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block3_conv1_layer_call_fn_6539620780inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block3_conv1_layer_call_and_return_conditional_losses_6539620796inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_block3_conv2_layer_call_fn_6539620810inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_block3_conv2_layer_call_and_return_conditional_losses_6539620826inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_block0_pool_layer_call_fn_6539620831inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539620839inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_block1_pool_layer_call_fn_6539620844inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539620852inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_block2_pool_layer_call_fn_6539620857inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539620865inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_block3_pool_layer_call_fn_6539620870inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539620878inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЭBш
3__inference_block0_dropout_layer_call_fn_6539620883inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
3__inference_block0_dropout_layer_call_fn_6539620888inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620893inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620905inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЭBш
3__inference_block1_dropout_layer_call_fn_6539620910inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
3__inference_block1_dropout_layer_call_fn_6539620915inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620920inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620932inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЭBш
3__inference_block2_dropout_layer_call_fn_6539620937inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
3__inference_block2_dropout_layer_call_fn_6539620942inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620947inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620959inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 г
K__inference_FeaturesVec_layer_call_and_return_conditional_losses_6539620586]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         @
џ ё
0__inference_FeaturesVec_layer_call_fn_6539620576P0б-
&б#
!і
inputs         ђ
ф "і         @е
%__inference__wrapped_model_65396109924б1
*б'
%і"
input_1         d
ф "3ф0
.
output_1"і
output_1         @┤
L__inference_block0_conv1_layer_call_and_return_conditional_losses_6539620616d3б0
)б&
$і!
inputs         c
ф ")б&
і
0         c 
џ ї
1__inference_block0_conv1_layer_call_fn_6539620600W3б0
)б&
$і!
inputs         c
ф "і         c ┤
L__inference_block0_conv2_layer_call_and_return_conditional_losses_6539620646d3б0
)б&
$і!
inputs         c 
ф ")б&
і
0         c 
џ ї
1__inference_block0_conv2_layer_call_fn_6539620630W3б0
)б&
$і!
inputs         c 
ф "і         c Х
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620893d7б4
-б*
$і!
inputs          
p 
ф ")б&
і
0          
џ Х
N__inference_block0_dropout_layer_call_and_return_conditional_losses_6539620905d7б4
-б*
$і!
inputs          
p
ф ")б&
і
0          
џ ј
3__inference_block0_dropout_layer_call_fn_6539620883W7б4
-б*
$і!
inputs          
p 
ф "і          ј
3__inference_block0_dropout_layer_call_fn_6539620888W7б4
-б*
$і!
inputs          
p
ф "і          н
K__inference_block0_pool_layer_call_and_return_conditional_losses_6539620839ёEбB
;б8
6і3
inputs'                           
ф ";б8
1і.
0'                           
џ Ф
0__inference_block0_pool_layer_call_fn_6539620831wEбB
;б8
6і3
inputs'                           
ф ".і+'                           ┤
L__inference_block1_conv1_layer_call_and_return_conditional_losses_6539620676d3б0
)б&
$і!
inputs          
ф ")б&
і
0         @
џ ї
1__inference_block1_conv1_layer_call_fn_6539620660W3б0
)б&
$і!
inputs          
ф "і         @┤
L__inference_block1_conv2_layer_call_and_return_conditional_losses_6539620706d3б0
)б&
$і!
inputs         @
ф ")б&
і
0         @
џ ї
1__inference_block1_conv2_layer_call_fn_6539620690W3б0
)б&
$і!
inputs         @
ф "і         @Х
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620920d7б4
-б*
$і!
inputs         @
p 
ф ")б&
і
0         @
џ Х
N__inference_block1_dropout_layer_call_and_return_conditional_losses_6539620932d7б4
-б*
$і!
inputs         @
p
ф ")б&
і
0         @
џ ј
3__inference_block1_dropout_layer_call_fn_6539620910W7б4
-б*
$і!
inputs         @
p 
ф "і         @ј
3__inference_block1_dropout_layer_call_fn_6539620915W7б4
-б*
$і!
inputs         @
p
ф "і         @н
K__inference_block1_pool_layer_call_and_return_conditional_losses_6539620852ёEбB
;б8
6і3
inputs'                           
ф ";б8
1і.
0'                           
џ Ф
0__inference_block1_pool_layer_call_fn_6539620844wEбB
;б8
6і3
inputs'                           
ф ".і+'                           х
L__inference_block2_conv1_layer_call_and_return_conditional_losses_6539620736e3б0
)б&
$і!
inputs         @
ф "*б'
 і
0         ђ
џ Ї
1__inference_block2_conv1_layer_call_fn_6539620720X3б0
)б&
$і!
inputs         @
ф "і         ђХ
L__inference_block2_conv2_layer_call_and_return_conditional_losses_6539620766f4б1
*б'
%і"
inputs         ђ
ф "*б'
 і
0         ђ
џ ј
1__inference_block2_conv2_layer_call_fn_6539620750Y4б1
*б'
%і"
inputs         ђ
ф "і         ђИ
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620947f8б5
.б+
%і"
inputs         ђ
p 
ф "*б'
 і
0         ђ
џ И
N__inference_block2_dropout_layer_call_and_return_conditional_losses_6539620959f8б5
.б+
%і"
inputs         ђ
p
ф "*б'
 і
0         ђ
џ љ
3__inference_block2_dropout_layer_call_fn_6539620937Y8б5
.б+
%і"
inputs         ђ
p 
ф "і         ђљ
3__inference_block2_dropout_layer_call_fn_6539620942Y8б5
.б+
%і"
inputs         ђ
p
ф "і         ђн
K__inference_block2_pool_layer_call_and_return_conditional_losses_6539620865ёEбB
;б8
6і3
inputs'                           
ф ";б8
1і.
0'                           
џ Ф
0__inference_block2_pool_layer_call_fn_6539620857wEбB
;б8
6і3
inputs'                           
ф ".і+'                           Х
L__inference_block3_conv1_layer_call_and_return_conditional_losses_6539620796f4б1
*б'
%і"
inputs         ђ
ф "*б'
 і
0         ђ
џ ј
1__inference_block3_conv1_layer_call_fn_6539620780Y4б1
*б'
%і"
inputs         ђ
ф "і         ђХ
L__inference_block3_conv2_layer_call_and_return_conditional_losses_6539620826f4б1
*б'
%і"
inputs         ђ
ф "*б'
 і
0         ђ
џ ј
1__inference_block3_conv2_layer_call_fn_6539620810Y4б1
*б'
%і"
inputs         ђ
ф "і         ђн
K__inference_block3_pool_layer_call_and_return_conditional_losses_6539620878ёEбB
;б8
6і3
inputs'                           
ф ";б8
1і.
0'                           
џ Ф
0__inference_block3_pool_layer_call_fn_6539620870wEбB
;б8
6і3
inputs'                           
ф ".і+'                           О
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539618936ЂDбA
*б'
%і"
input_1         d
ф

trainingp "%б"
і
0         @
џ О
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539619069ЂDбA
*б'
%і"
input_1         d
ф

trainingp"%б"
і
0         @
џ н
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620397Bб?
(б%
#і 
input         d
ф

trainingp "%б"
і
0         @
џ н
Q__inference_deep_coffea_model_layer_call_and_return_conditional_losses_6539620554Bб?
(б%
#і 
input         d
ф

trainingp"%б"
і
0         @
џ «
6__inference_deep_coffea_model_layer_call_fn_6539615925tDбA
*б'
%і"
input_1         d
ф

trainingp "і         @«
6__inference_deep_coffea_model_layer_call_fn_6539618815tDбA
*б'
%і"
input_1         d
ф

trainingp"і         @г
6__inference_deep_coffea_model_layer_call_fn_6539620150rBб?
(б%
#і 
input         d
ф

trainingp "і         @г
6__inference_deep_coffea_model_layer_call_fn_6539620261rBб?
(б%
#і 
input         d
ф

trainingp"і         @Ф
I__inference_flatten_1_layer_call_and_return_conditional_losses_6539620565^4б1
*б'
%і"
inputs         ђ
ф "&б#
і
0         ђ
џ Ѓ
.__inference_flatten_1_layer_call_fn_6539620559Q4б1
*б'
%і"
inputs         ђ
ф "і         ђи
(__inference_signature_wrapper_6539620051і?б<
б 
5ф2
0
input_1%і"
input_1         d"3ф0
.
output_1"і
output_1         @