class ColleyQueries:
    ### REQUEST_QUERIES #######################################################
    ## [조건] 사용자
    # 삭제안함
    SQL_USER_LIST = """
    SELECT user_id 
    FROM colley_v01.user_list;
    """

    # [게시글] 게시글 식별자, 게시글 작성자, 제목, 본문, 태그목록, 판매요청 활성화 여부, 게시된 시간, 변경된 시간, 게시글 노출 여부
    SQL_BOARD_LIST = """
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
    # [상품] 상품 식별자, 상품명, 태그목록, 라이센스 명, 라이센스 식별자, 카테고리 식별자
    SQL_PRODUCT_LIST = """
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
        AND license_id != 0
        AND category_id != 0);
    """

    SQL_BOARD_PRODUCT_LIST = """
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

    ## [태그] 사용자관심태그, 게시글 태그, 상품 태그, 라이센스, 카테고리
    # 사용자관심태그
    SQL_USER_INTEREST_TAG_LIST = """
    SELECT 
        user_id,
        tag 
    FROM colley_v01.interest_tag_list
    WHERE tag_id != 0;
    """
    # 게시글 태그
    SQL_BOARD_TAG_LIST = """
    SELECT 
        item_id AS board_id,
        tag
    FROM colley_v01.item_tag_list
    WHERE tag_id != 0;
    """
    # 상품 태그
    SQL_PRODUCT_TAG_LIST = """
    SELECT 
        product_id,
        tag
    FROM colley_v01.product_tag_list
    WHERE tag_id != 0;
    """
    # 라이센스
    SQL_LICENSE_TAG_LIST = """
    SELECT
        license_id,
        license_name,
        parent_license_id
    FROM colley_v01.license_list;
    """
    # 카테고리
    SQL_CATEGORY_TAG_LIST = """
    SELECT
        category_id,
        category_name,
        parent_category_id
    FROM colley_v01.category_list;
    """

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
        AND pdl.license_id != 0
        AND pdl.category_id != 0);
    """
    # 게시글의 상품연계 가능시, 게시자를 구매내역에 추가
    SQL_OWNED_USER_LIST = """
    SELECT 
        il.user_id AS user_id,
        ipl.product_id AS product_id,
        il.created_time AS created_time
    FROM colley_v01.item_list AS il
    JOIN colley_v01.item_product_list AS ipl
        ON il.item_id = ipl.item_id
        JOIN colley_v01.product_list AS pl
            ON ipl.product_id = pl.product_id
    WHERE (
        ipl.product_id != 0
        AND TRIM(pl.tag_string) != ''
        AND pl.license_id != 0
        AND pl.category_id != 0
    );
    """
    # 판매요청 조회 후, 상품연계 가능시 해당 사용자들을 구매내역에 추가, 연계불가 시 좋아요에 추가.
    SQL_REQUEST_LIST = """
    SELECT 
        sl.user_id AS user_id,
        ipl.product_id AS product_id,
        sl.created_time AS created_time
    FROM colley_v01.interaction_sell_request_list AS sl
    JOIN colley_v01.item_product_list AS ipl
        ON sl.item_id = ipl.item_id
        JOIN colley_v01.product_list AS pl
            ON ipl.product_id = pl.product_id
    WHERE (
        ipl.product_id != 0
        AND TRIM(pl.tag_string) != ''
        AND pl.license_id != 0
        AND pl.category_id != 0
    );
    """
    # 스크랩 내용 조회 후, 상품연계 가능 시 해당 사용자들을 구매내역에 추가, 연계불가 시 좋아요에 추가.
    # 이거... target_type의미를 알아야 연계할 수 있을 듯..... 8개???????
    # 뭔지 모르겠음.. 일단 PASS

    ## [열람] 사용자 식별자, 게시글|상품 식별자, 의사결정시간(datetime)
    # 게시글
    SQL_OPEN_BOARD_LIST = """
    SELECT 
        user_id,
        item_id AS board_id,
        created_time
    FROM colley_v01.open_item_list;
    """
    # 상품
    SQL_OPEN_PRODUCT_LIST = """
    SELECT
        user_id,
        product_id,
        created_time
    FROM colley_v01.open_product_list;
    """
    ## [좋아요] 사용자 식별자, 게시글|상품 식별자, 의사결정시간(datetime): 게시글, 상품
    # 게시글
    SQL_LIKE_BOARD_LIST = """
    SELECT
        user_id,
        item_id AS board_id,
        created_time
    FROM colley_v01.interaction_like_list;
    """
    # 상품: 상품의 좋아요가 없으니, 게시글과의 연계로 좋아요 내역생성
    SQL_LIKE_PRODUCT_LIST = """
    SELECT
        ill.user_id AS user_id,
        ipl.product_id AS product_id,
        ill.created_time AS created_time
    FROM colley_v01.interaction_like_list AS ill
    JOIN colley_v01.item_product_list AS ipl
        ON ill.item_id = ipl.item_id
    WHERE ipl.product_id != 0;
    """

    SQL_DEFINED_TAG_LIST = """
    SELECT 
        tag_id,
        tag
    FROM colley_v01.tag_list;
    """

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

    ### REQUEST_QUERIES #######################################################
