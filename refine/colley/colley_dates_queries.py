class ColleyDatesQueries:
    ### REQUEST_QUERIES #######################################################
    ## [조건] 사용자
    def __init__(
        self,
        begin_date_str: str,
        emit_date_str: str,
    ) -> None:
        ##
        self.__redef_request_queries_str__(
            begin_date_str,
            emit_date_str,
        )

    # end : init()

    def __redef_request_queries_str__(
        self,
        begin_date_str: str,
        emit_date_str: str,
    ) -> None:
        """질의문을 재작성합니다."""
        self.SQL_PURCHASE_LIST = str.format(
            self.SQL_PURCHASE_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_OWNED_USER_LIST = str.format(
            self.SQL_OWNED_USER_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_BOARD_REQUEST_LIST = str.format(
            self.SQL_BOARD_REQUEST_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_OPEN_BOARD_LIST = str.format(
            self.SQL_OPEN_BOARD_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_OPEN_PRODUCT_LIST = str.format(
            self.SQL_OPEN_PRODUCT_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_LIKE_BOARD_LIST = str.format(
            self.SQL_LIKE_BOARD_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_LIKE_PRODUCT_LIST = str.format(
            self.SQL_LIKE_PRODUCT_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_BOARD_SCRAP_LIST = str.format(
            self.SQL_BOARD_SCRAP_LIST,
            begin_date_str,
            emit_date_str,
        )
        self.SQL_PRODUCT_SCRAP_LIST = str.format(
            self.SQL_PRODUCT_SCRAP_LIST,
            begin_date_str,
            emit_date_str,
        )

    ### [FORALL] REQUEST METADATA QUERIES
    # 삭제안함
    SQL_ALL_USER_LIST = """
    SELECT user_id 
    FROM colley_v01.user_list;
    """
    """user_id"""

    # [게시글] 게시글 식별자, 게시글 작성자, 제목, 본문, 태그목록, 판매요청 활성화 여부, 게시된 시간, 변경된 시간, 게시글 노출 여부
    SQL_ALL_BOARD_LIST = """
    SELECT 
        item_id AS board_id,
        user_id, 
        title, 
        content, 
        tag_string, 
        sell_request_available, 
        created_time,
        modified_time,
        is_listed
    FROM colley_v01.item_list
    WHERE TRIM(tag_string) != '';
    """
    """board_id, user_id, title, content, tag_string, sell_request_available, created_time, modified_time, is_listed"""

    # [상품] 상품 식별자, 상품명, 태그목록, 라이센스 명, 라이센스 식별자, 카테고리 식별자
    SQL_ALL_PRODUCT_LIST = """
    SELECT 
        product_id, 
        product_name,
        tag_string, 
        license_name, 
        license_id, 
        category_id
    FROM colley_v01.product_list
    WHERE (
        TRIM(tag_string) != ''
        # AND license_id != 0
        # AND category_id != 0
    );
    """
    """product_id, product_name, tag_string, license_name, license_id, category_id"""

    SQL_ALL_BOARD_PRODUCT_LIST = """
    SELECT 
        ipl.item_id AS board_id,
        ipl.product_id AS product_id,
        ipl.product_name AS product_name,
        pl.tag_string AS tag_string,
        pl.license_id AS license_id,
        pl.category_id AS category_id
    FROM colley_v01.item_product_list AS ipl
    JOIN colley_v01.item_list AS il
        ON ipl.item_id = il.item_id
        JOIN colley_v01.product_list AS pl
            ON ipl.product_id = pl.product_id
    WHERE (
        ipl.product_id != 0
        AND TRIM(pl.tag_string) != ''
        AND pl.license_id != 0
        AND pl.category_id != 0);
    """
    """board_id, product_id, product_name, tag_string, license_id, category_id"""

    ## [태그] 사용자관심태그, 게시글 태그, 상품 태그, 라이센스, 카테고리
    # 사용자관심태그
    SQL_ALL_USER_INTEREST_TAG_LIST = """
    SELECT 
        user_id,
        tag,
        tag_id
    FROM colley_v01.interest_tag_list
    WHERE tag_id != 0;
    """
    """user_id, tag, tag_id"""

    # 게시글 태그
    SQL_ALL_BOARD_TAG_LIST = """
    SELECT 
        il.item_id AS board_id,
        itl.tag AS tag,
        itl.tag_id AS tag_id
    FROM colley_v01.item_list AS il
        JOIN colley_v01.item_tag_list AS itl
        ON il.item_id = itl.item_id
    WHERE (
            itl.tag_id != 0
            AND itl.tag NOT LIKE '%#%'
            AND itl.tag NOT LIKE '%챌린지%'
        );
    """
    """board_id, tag, tag_id"""

    # 상품 태그
    SQL_ALL_PRODUCT_TAG_LIST = """
    SELECT
        ppl.product_id AS product_id,
        ptl.tag AS tag,
        ptl.tag_id AS tag_id
    FROM colley_v01.purchase_list AS pl
    JOIN colley_v01.purchase_product_list AS ppl
        ON pl.purchase_id = ppl.purchase_id
        JOIN colley_v01.product_list AS pr
            ON ppl.product_id = pr.product_id
            JOIN colley_v01.product_tag_list AS ptl
                ON pr.product_id = ptl.product_id
    WHERE (
            ptl.tag_id != 0
            AND ptl.tag NOT LIKE '%#%'
            AND ptl.tag NOT LIKE '%챌린지%'
    )
    ORDER BY ppl.product_id ASC;
    """
    """product_id, tag, tag_id"""

    ### [SUBSET] DECISIONS OF CONDITIONAL REQUEST QUERIES (VIEWS, LIKES, PURCHASES)
    ## [구매내역] 사용자 식별자, 상품 식별자, 의사결정시간(datetime)
    # 구매내역과 상품내역을 연계해서 상품구매목록 생성
    SQL_PURCHASE_LIST = """
    SELECT
        pl.user_id AS user_id,
        ppl.product_id AS product_id,
        pl.created_time AS created_time
    FROM colley_v01.purchase_list AS pl
    JOIN colley_v01.purchase_product_list AS ppl
        ON pl.purchase_id = ppl.purchase_id
        JOIN colley_v01.product_list AS pdl
            ON ppl.product_id = pdl.product_id
    WHERE (
        TRIM(pdl.tag_string) != ''
        # AND pdl.license_id != 0
        # AND pdl.category_id != 0
        ) AND (pl.created_time BETWEEN '{0}' AND '{1}');
    """
    """user_id, product_id, created_time"""

    # 게시글의 상품연계 가능시, 게시자를 구매내역에 추가
    SQL_OWNED_USER_LIST = """
    SELECT 
        il.user_id AS user_id,
        il.item_id AS board_id,
        ipl.product_id AS product_id,
        il.created_time AS created_time
    FROM colley_v01.item_list AS il
    JOIN colley_v01.item_product_list AS ipl
        ON il.item_id = ipl.item_id
        JOIN colley_v01.product_list AS pl
            ON ipl.product_id = pl.product_id
    WHERE (
        TRIM(pl.tag_string) != ''
        # ipl.product_id != 0
        # AND TRIM(pl.tag_string) != ''
        # AND pl.license_id != 0
        # AND pl.category_id != 0
    )   AND (il.created_time BETWEEN '{0}' AND '{1}');
    """
    """user_id, board_id, product_id, created_time"""

    # 판매요청 조회 후, 상품연계 가능시 해당 사용자들을 구매내역에 추가, 연계불가 시 좋아요에 추가.
    SQL_BOARD_REQUEST_LIST = """
    SELECT 
        sl.user_id AS user_id,
        sl.item_id AS board_id,
        ipl.product_id AS product_id,
        sl.created_time AS created_time
    FROM colley_v01.interaction_sell_request_list AS sl
    JOIN colley_v01.item_product_list AS ipl
        ON sl.item_id = ipl.item_id
        JOIN colley_v01.product_list AS pl
            ON ipl.product_id = pl.product_id
    WHERE (
        TRIM(pl.tag_string) != ''
        # ipl.product_id != 0
        # AND TRIM(pl.tag_string) != ''
        # AND pl.license_id != 0
        # AND pl.category_id != 0
    )   AND (sl.created_time BETWEEN '{0}' AND '{1}');
    """
    """user_id, board_id, product_id, created_time"""

    SQL_BOARD_SCRAP_LIST = """
    SELECT
        isl.user_id AS user_id,
        isl.target_id AS board_id,
        ipl.product_id AS product_id,
        isl.created_time AS created_time
    FROM colley_v01.interaction_scrap_list AS isl
    JOIN colley_v01.item_list AS il
        ON isl.target_id = il.item_id
        JOIN colley_v01.item_product_list AS ipl
            ON isl.target_id = ipl.item_id
    WHERE isl.target_type = 1
        AND (isl.created_time BETWEEN '{0}' AND '{1}');
    """
    """user_id, board_id, product_id, created_time"""

    SQL_PRODUCT_SCRAP_LIST = """
    SELECT
        isl.user_id AS user_id,
        isl.target_id AS product_id,
        isl.created_time AS created_time
    FROM colley_v01.interaction_scrap_list AS isl
    JOIN colley_v01.product_list AS pl
        ON isl.target_id = pl.product_id
    WHERE target_type = 4
        AND (isl.created_time BETWEEN '{0}' AND '{1}');
    """
    """user_id, product_id, created_time"""

    ## [좋아요] 사용자 식별자, 게시글|상품 식별자, 의사결정시간(datetime): 게시글, 상품
    # 게시글
    SQL_LIKE_BOARD_LIST = """
    SELECT
        user_id,
        item_id AS board_id,
        created_time
    FROM colley_v01.interaction_like_list
    WHERE created_time BETWEEN '{0}' AND '{1}';
    """
    """user_id, board_id, created_time"""
    # 상품: 상품의 좋아요가 없으니, 게시글과의 연계로 좋아요 내역생성

    SQL_LIKE_PRODUCT_LIST = """
    SELECT
        ill.user_id AS user_id,
        ipl.product_id AS product_id,
        ill.created_time AS created_time
    FROM colley_v01.interaction_like_list AS ill
    JOIN colley_v01.item_product_list AS ipl
        ON ill.item_id = ipl.item_id
    WHERE ipl.product_id != 0
        AND (ill.created_time BETWEEN '{0}' AND '{1}');
    """
    """
    - 이 내역을 통합할 때, 중복된 항목들이 만들어지는 것 같으니, 재처리 과정에서 확인할 것
        - user_id, product_id, created_time
    """

    ## [열람] 사용자 식별자, 게시글|상품 식별자, 의사결정시간(datetime)
    # 게시글
    SQL_OPEN_BOARD_LIST = """
    SELECT 
        user_id,
        item_id AS board_id,
        created_time
    FROM colley_v01.open_item_list
    WHERE created_time BETWEEN '{0}' AND '{1}';
    """
    """user_id, board_id, created_time"""
    # 상품
    SQL_OPEN_PRODUCT_LIST = """
    SELECT
        user_id,
        product_id,
        created_time
    FROM colley_v01.open_product_list
    WHERE created_time BETWEEN '{0}' AND '{1}';
    """
    """user_id, product_id, created_time"""

    '''
    SQL_PRODUCT_SCRAP_LIST = """
    SELECT
        user_id,
        target_id AS product_id,
        created_time
    FROM colley_v01.interaction_scrap_list
    WHERE target_type = type_id;
    """
    """
    - target_type을 변경해서 사용하세요.
        - ex. 게시글을 가져오려면 SQL_PRODUCT_SCRAP_LIST.replace("type_id", "1")한 후, 호출하세요.
    - type info.
        - 1: 아이템
        - 2: 컬렉션
        - 3: 아티클 (현재 안쓰임)
        - 4: 콜리샵
        - 5: 콜리콘텐츠 (현재 안쓰임)
        - 6: 덕친마켓
        - 7: 옥션 (안쓰임)
        - 8: 콜리포스트
    """
    '''
    ### REQUEST_QUERIES #######################################################
