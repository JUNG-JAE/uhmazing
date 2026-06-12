"""데이터셋 이름 -> Dataset 클래스 매핑 및 DataLoader 생성 팩토리."""

from data_provider.data_loader import Dataset_ETT_hour, Dataset_hanwoo, hanwoo_collate
from torch.utils.data import DataLoader

# 데이터셋 이름과 해당 Dataset 클래스 매핑
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'hanwoo': Dataset_hanwoo,
}


def data_provider(args, flag):
    """
    flag('train'|'val'|'test')에 맞는 Dataset과 DataLoader를 생성해 반환한다.
    """
    # ---- 한우(분류) 전용 경로: 시계열+프롬프트 샘플, 커스텀 collate ----
    if args.data == 'hanwoo':
        shuffle_flag = (flag == 'train')          # test/val은 셔플 안 함
        drop_last = (flag == 'train')
        data_set = Dataset_hanwoo(
            root_path=args.root_path, flag=flag, file_type=args.file_type,
            weather_interval=args.weather_interval, weather_mode=args.weather_mode,
            weather_months=args.weather_months)
        data_loader = DataLoader(
            data_set, batch_size=args.batch_size, shuffle=shuffle_flag,
            num_workers=args.num_workers, drop_last=drop_last, collate_fn=hanwoo_collate)
        return data_set, data_loader

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1   # timeF면 연속 시간 특성 사용
    percent = args.percent

    # test는 셔플하지 않는다. (학습/검증은 셔플)
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    
    return data_set, data_loader
