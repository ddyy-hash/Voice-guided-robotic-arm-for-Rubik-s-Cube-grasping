from voice_rubik_grasping.voice import VoiceCommandParser


def test_parser_maps_rubik_cube_command():
    parser = VoiceCommandParser("config/voice_aliases_zh.yaml")

    command = parser.parse("请帮我抓取魔方")

    assert command.target == "cube"
    assert command.confidence >= 0.7
    assert not command.is_exit


def test_parser_combines_color_and_object():
    parser = VoiceCommandParser("config/voice_aliases_zh.yaml")

    command = parser.parse("抓红色杯子")

    assert command.target == "red cup"


def test_parser_recognizes_exit_command():
    parser = VoiceCommandParser("config/voice_aliases_zh.yaml")

    command = parser.parse("停止")

    assert command.is_exit
