@testset "trading calendar" begin
    @testset "weekends are skipped" begin
        @test trading_days(Date(2026, 1, 2), 3) ==
            [Date(2026, 1, 2), Date(2026, 1, 5), Date(2026, 1, 6)]
    end

    @testset "a weekend start rolls forward" begin
        @test first(trading_days(Date(2026, 1, 3), 1)) == Date(2026, 1, 5)
    end

    @testset "days are strictly increasing weekdays" begin
        days = trading_days(Date(2026, 1, 1), 60)
        @test length(days) == 60
        @test issorted(days) && allunique(days)
        @test all(is_trading_day, days)
    end

    @testset "degenerate counts" begin
        @test isempty(trading_days(Date(2026, 1, 1), 0))
        @test_throws ArgumentError trading_days(Date(2026, 1, 1), -1)
    end

    @testset "a daily bar is stamped at the session close" begin
        @test session_close(Date(2026, 1, 2)) == DateTime(2026, 1, 2, 10, 0)
    end

    @testset "the annualisation convention is fixed" begin
        @test BARS_PER_YEAR == 252
        @test ANNUALISER ≈ sqrt(252)
        @test annualise(deannualise(0.3)) ≈ 0.3
        @test deannualise(0.252) ≈ 0.252 / sqrt(252)
    end
end

@testset "exchange calendar" begin
    @testset "a holiday is not a trading day and a weekday is" begin
        calendar = nse_calendar()
        @test !is_trading_day(calendar, Date(2026, 12, 25))    # Christmas, a Friday
        @test !is_trading_day(calendar, Date(2026, 1, 26))     # Republic Day, a Monday
        @test is_trading_day(calendar, Date(2026, 12, 24))
        @test !is_trading_day(calendar, Date(2026, 12, 26))    # a Saturday
        @test occursin("TradingCalendar", sprint(show, calendar))
    end

    @testset "every shipped holiday falls on a weekday" begin
        # A holiday list that puts a closure on a Sunday is a list that was transcribed from a
        # page listing weekend dates too, and it would then be silently short of real ones.
        for date in vcat(NSE_HOLIDAYS_2025, NSE_HOLIDAYS_2026)
            @test dayofweek(date) <= 5
        end
        @test allunique(NSE_HOLIDAYS_2025)
        @test allunique(NSE_HOLIDAYS_2026)
        @test length(NSE_HOLIDAYS_2025) == 14
        @test length(NSE_HOLIDAYS_2026) == 16

        # 2026: 261 weekdays, 16 of them closed, one Sunday session.
        calendar = nse_calendar()
        @test count(
            day -> is_trading_day(calendar, day),
            Date(2026, 1, 1):Day(1):Date(2026, 12, 31),
        ) == 246
    end

    @testset "a muhurat session trades but is not a full day" begin
        # One hour on a day the regular session is shut, and on a Sunday in 2026. Its bar has a
        # day's stamp and an hour's volume, so counting it as an ordinary observation pulls
        # realised volatility down.
        calendar = nse_calendar()
        for date in NSE_MUHURAT
            @test is_trading_day(calendar, date)
            @test !is_full_session(calendar, date)
        end
        @test dayofweek(Date(2026, 11, 8)) == 7                # a Sunday that trades
        @test Date(2025, 10, 21) in NSE_HOLIDAYS_2025          # closed and open at once
        @test is_full_session(calendar, Date(2026, 12, 24))
        @test !is_full_session(calendar, Date(2026, 12, 25))
        @test !is_full_session(calendar, Date(2026, 12, 26))
    end

    @testset "a year it does not cover is refused, not guessed" begin
        # A year with nothing listed looks exactly like a year nobody entered, and the
        # arithmetic returns a confident wrong answer in both cases.
        calendar = nse_calendar()
        @test covers(calendar, Date(2026, 6, 1))
        @test !covers(calendar, Date(2024, 6, 1))
        @test_throws CalendarCoverageError is_trading_day(calendar, Date(2024, 6, 3))
        @test_throws CalendarCoverageError is_trading_day(calendar, Date(2027, 6, 3))
        @test_throws CalendarCoverageError is_full_session(calendar, Date(2024, 6, 3))
        rendered = sprint(
            showerror, CalendarCoverageError(2024, 2025:2026),
        )
        @test occursin("2024", rendered)
        @test occursin("2025", rendered)
    end

    @testset "a calendar cannot claim less than it lists" begin
        @test_throws ArgumentError TradingCalendar(
            [Date(2025, 1, 1)]; covered = 2026:2026,
        )
        @test_throws ArgumentError TradingCalendar(Date[])
        @test TradingCalendar(Date[]; covered = 2030:2030).covered == 2030:2030
        derived = TradingCalendar([Date(2025, 5, 1), Date(2027, 5, 1)])
        @test derived.covered == 2025:2027
    end

    @testset "trading days over a calendar skip the holidays" begin
        calendar = nse_calendar()
        days = trading_days(calendar, Date(2026, 12, 23), 3)
        @test days == [Date(2026, 12, 23), Date(2026, 12, 24), Date(2026, 12, 28)]
        @test isempty(trading_days(calendar, Date(2026, 1, 1), 0))
        @test_throws ArgumentError trading_days(calendar, Date(2026, 1, 1), -1)
    end

    @testset "a calendar can be supplied for a year that is not shipped" begin
        # Holidays are published yearly and this package ships only what it has checked.
        mktempdir() do dir
            path = joinpath(dir, "holidays.txt")
            write(
                path,
                """
                # exchange holidays, one per line
                2027-01-26   # Republic Day

                2027-12-25
                """,
            )
            calendar = load_calendar(path; covered = 2027:2027)
            @test !is_trading_day(calendar, Date(2027, 1, 26))
            @test is_trading_day(calendar, Date(2027, 1, 27))
            @test length(calendar.holidays) == 2

            write(path, "not-a-date\n")
            @test_throws ArgumentError load_calendar(path; covered = 2027:2027)
            @test_throws ArgumentError load_calendar(
                joinpath(dir, "absent.txt"); covered = 2027:2027,
            )
        end
    end

    @testset "a holiday break is not a hole in the download" begin
        # Without a calendar this is the check that cries wolf every October, and the usual
        # response is to raise the threshold until it stops noticing real outages.
        bars = Bar[
            Bar("X", session_close(day), 100.0, 101.0, 99.0, 100.0, 1000.0)
                for day in Date(2026, 12, 18):Day(1):Date(2026, 12, 31)
                if is_trading_day(nse_calendar(), day)
        ]
        naive = validate_bars(bars; min_bars = 5, max_gap_bars = 0)
        informed = validate_bars(
            bars; calendar = nse_calendar(), min_bars = 5, max_gap_bars = 0,
        )
        gap_issues(report) = [
            issue for issue in report.issues if issue.check === :calendar_gap
        ]
        @test !isempty(gap_issues(naive))       # Christmas looks like a missing day
        @test isempty(gap_issues(informed))

        @test BayesTrade.trading_days_between(
            DateTime(2026, 12, 24, 10), DateTime(2026, 12, 28, 10), nse_calendar(),
        ) == 0
        @test BayesTrade.trading_days_between(
            DateTime(2026, 12, 24, 10), DateTime(2026, 12, 28, 10),
        ) == 1                                   # the 25th, counted as an ordinary Friday

        # A series running past the end of the calendar keeps reporting rather than raising.
        @test BayesTrade.trading_days_between(
            DateTime(2027, 3, 1, 10), DateTime(2027, 3, 8, 10), nse_calendar(),
        ) == 4
    end
end
