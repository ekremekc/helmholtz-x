source = GetActiveSource()

all_pieces = [ source ]

realCalculator=servermanager.Fetch(source)
rcPointData=realCalculator.GetPointData()

Result2Array=rcPointData.GetArray('real_P_1_Direct')
#carpim = Result2Array*2

for i in range(1, 16):
    transform = Transform(Input=source)
    transform.Transform.Rotate = [ 0, 0, 22.5 * i ]
    all_pieces.append(transform)

group = GroupDatasets(Input=all_pieces)
merge = MergeBlocks(Input=group)

Hide(source)
Show(merge)
