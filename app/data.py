"""네이버 뉴스 카테고리별 기사 제목 수집 모듈.

방법 1: 내장 샘플 데이터 (인터넷 없이 실행 가능)
방법 2: 네이버 뉴스 페이지에서 실시간 기사 제목 스크래핑
"""

from __future__ import annotations

from typing import List, Tuple

# ▼▼▼ [추가] 스크래핑용 라이브러리 임포트 ▼▼▼
# 기존: 임포트 없음 (내장 샘플만 사용)
# 변경: requests + BeautifulSoup 을 이용한 네이버 뉴스 실시간 수집 기능 추가
import requests
from bs4 import BeautifulSoup
# ▲▲▲ [추가] 스크래핑용 라이브러리 임포트 ▲▲▲


# ────────────────────────────────────────────────────────────
# 방법 1: 내장 샘플 데이터
# ▼▼▼ [변경] 데이터 전면 교체 ▼▼▼
# 기존: 영어 BBC 기사 5개 × 5카테고리 = 25건 (sport/business/politics/tech/entertainment)
# 변경: 한국어 네이버 뉴스 15개 × 3카테고리 = 45건 (it/sports/entertainment)
#        데이터 수 증가 → 학습 안정성과 정확도 향상에 기여
# ────────────────────────────────────────────────────────────
SAMPLE_DATA: List[Tuple[str, str]] = [
    # IT / 과학
    ("삼성전자 갤럭시 신제품 발표 AI 기능 대폭 강화", "it"),
    ("애플 아이폰 신형 출시 국내 예약 판매 시작", "it"),
    ("구글 딥마인드 새로운 인공지능 모델 공개 성능 향상", "it"),
    ("네이버 클로바 초거대 언어 모델 업그레이드 발표", "it"),
    ("카카오 자율주행 소프트웨어 도로 주행 테스트 통과", "it"),
    ("마이크로소프트 클라우드 서버 보안 취약점 패치 배포", "it"),
    ("SK하이닉스 차세대 반도체 메모리 양산 성공 발표", "it"),
    ("메타 가상현실 헤드셋 신제품 글로벌 출시 예정", "it"),
    ("엔비디아 그래픽카드 AI 학습용 신제품 공개 가격 확정", "it"),
    ("국내 스타트업 챗봇 서비스 글로벌 투자 유치 성공", "it"),
    ("테슬라 자율주행 소프트웨어 국내 도입 승인 완료", "it"),
    ("LG전자 스마트홈 플랫폼 새 기능 업데이트 출시", "it"),
    ("인텔 최신 프로세서 벤치마크 성능 기록 경신", "it"),
    ("국내 핀테크 기업 블록체인 결제 서비스 출시", "it"),
    ("쿠팡 물류 로봇 도입 배송 속도 대폭 단축 발표", "it"),

    # 스포츠
    ("손흥민 프리미어리그 골 기록 경신 팀 승리 이끌어", "sports"),
    ("류현진 메이저리그 복귀전 호투 팀 승리 기여", "sports"),
    ("국가대표 축구팀 월드컵 예선 승리 16강 진출 확정", "sports"),
    ("KBO 리그 한화이글스 연승 행진 순위 상승 중", "sports"),
    ("프로야구 포스트시즌 한국시리즈 대진 확정 발표", "sports"),
    ("이강인 파리생제르맹 선발 출전 어시스트 기록", "sports"),
    ("황희찬 울버햄튼 연속골 프리미어리그 공격 선두", "sports"),
    ("한국 배드민턴 대표팀 아시아선수권 금메달 획득", "sports"),
    ("국내 프로농구 챔피언십 결승 티켓 두 팀 확정", "sports"),
    ("여자 골프 고진영 미국투어 우승 통산 트로피 추가", "sports"),
    ("한국 수영 국가대표 세계선수권 결선 진출 성공", "sports"),
    ("전국체육대회 개막 지역 대표 선수들 열띤 경쟁", "sports"),
    ("K리그 올시즌 득점왕 후보 세 선수 치열한 경쟁", "sports"),
    ("국제 마라톤 대회 한국 선수 개인 기록 갱신 완주", "sports"),
    ("올림픽 국가대표 선발전 각 종목 치열한 경쟁 펼쳐", "sports"),

    # 연예
    ("BTS 새 앨범 전 세계 음원 차트 동시 1위 달성", "entertainment"),
    ("블랙핑크 월드투어 한국 공연 티켓 조기 매진", "entertainment"),
    ("넷플릭스 한국 드라마 오징어게임 시즌2 예고편 공개", "entertainment"),
    ("영화 범죄도시 속편 개봉 첫날 관객 백만 돌파", "entertainment"),
    ("아이유 신곡 발매 실시간 차트 전 사이트 1위", "entertainment"),
    ("뉴진스 일본 데뷔 쇼케이스 현지 팬들 열렬한 환호", "entertainment"),
    ("세븐틴 유럽 공연 매진 케이팝 인기 다시 증명", "entertainment"),
    ("배우 박서준 할리우드 신작 영화 캐스팅 확정 발표", "entertainment"),
    ("스트레이키즈 미국 빌보드 앨범 차트 상위권 진입", "entertainment"),
    ("한국 영화 칸 국제영화제 경쟁 부문 초청 확정", "entertainment"),
    ("드라마 이상한 변호사 우영우 해외 OTT 동시 방영", "entertainment"),
    ("케이팝 아이돌 그룹 팬미팅 전국 투어 일정 공개", "entertainment"),
    ("트와이스 컴백 쇼케이스 온라인 동시 접속 기록 경신", "entertainment"),
    ("배우 송혜교 새 드라마 주연 캐스팅 최종 확정", "entertainment"),
    ("국내 웹툰 원작 드라마 해외 수출 계약 대규모 성사", "entertainment"),
]
# ▲▲▲ [변경] 데이터 전면 교체 ▲▲▲


# ────────────────────────────────────────────────────────────
# ▼▼▼ [추가] 방법 2: 네이버 뉴스 실시간 스크래핑 함수 ▼▼▼
# 기존: 스크래핑 기능 없음
# 변경: 네이버 뉴스 IT(105), 스포츠(107), 연예(106) 섹션에서 기사 제목 수집
# ────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _scrape_naver_section(section_id: int, label: str, max_items: int = 20) -> List[Tuple[str, str]]:
    """네이버 뉴스 섹션 페이지에서 기사 제목을 수집한다.

    Args:
        section_id: 네이버 뉴스 섹션 번호 (105=IT, 107=스포츠, 106=연예)
        label: 반환할 튜플의 카테고리 라벨
        max_items: 최대 수집 건수

    Returns:
        (기사제목, 라벨) 튜플 목록. 오류 시 빈 리스트.
    """
    url = f"https://news.naver.com/section/{section_id}"
    try:
        response = requests.get(url, headers=_HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title_tags = soup.select("a.sa_text_title strong.sa_text_strong")
        results = []
        for tag in title_tags[:max_items]:
            title = tag.get_text(strip=True)
            if title:
                results.append((title, label))
        return results
    except Exception:
        return []


def get_news_it(max_items: int = 20) -> List[Tuple[str, str]]:
    """네이버 뉴스 IT/과학 섹션(105)에서 기사 제목을 가져온다."""
    return _scrape_naver_section(section_id=105, label="it", max_items=max_items)


def get_news_sports(max_items: int = 20) -> List[Tuple[str, str]]:
    """네이버 뉴스 스포츠 섹션(107)에서 기사 제목을 가져온다."""
    return _scrape_naver_section(section_id=107, label="sports", max_items=max_items)


def get_news_entertainment(max_items: int = 20) -> List[Tuple[str, str]]:
    """네이버 뉴스 연예 섹션(106)에서 기사 제목을 가져온다."""
    return _scrape_naver_section(section_id=106, label="entertainment", max_items=max_items)
# ▲▲▲ [추가] 방법 2: 네이버 뉴스 실시간 스크래핑 함수 ▲▲▲


def load_sample_data(use_scraping: bool = False) -> Tuple[List[str], List[str]]:
    """기사 제목 목록과 카테고리 라벨 목록을 반환한다.

    # ▼▼▼ [변경] use_scraping 파라미터 추가 ▼▼▼
    # 기존: 항상 내장 영어 데이터 반환
    # 변경: use_scraping=True 시 네이버 실시간 수집 → 실패하면 내장 데이터로 자동 대체(fallback)
    # ▲▲▲ [변경] use_scraping 파라미터 추가 ▲▲▲

    Args:
        use_scraping: True 이면 네이버 뉴스를 실시간으로 수집하고,
                      수집 결과가 부족하면 내장 샘플 데이터로 대체한다.
                      False(기본값)이면 내장 샘플 데이터를 사용한다.
    """
    if use_scraping:
        scraped: List[Tuple[str, str]] = []
        scraped.extend(get_news_it())
        scraped.extend(get_news_sports())
        scraped.extend(get_news_entertainment())
        data = scraped if len(scraped) >= 10 else SAMPLE_DATA
    else:
        data = SAMPLE_DATA

    texts = [text for text, _ in data]
    labels = [label for _, label in data]
    return texts, labels
