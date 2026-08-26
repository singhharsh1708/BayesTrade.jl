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
        fixture = ["a test fixture"]
        @test_throws ArgumentError TradingCalendar(
            [Date(2025, 1, 1)]; covered = 2026:2026, sources = fixture,
        )
        @test_throws ArgumentError TradingCalendar(Date[]; sources = fixture)
        @test TradingCalendar(
            Date[]; covered = 2030:2030, sources = fixture,
        ).covered == 2030:2030
        derived = TradingCalendar(
            [Date(2025, 5, 1), Date(2027, 5, 1)]; sources = fixture,
        )
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
            fixture = ["a test fixture"]
            calendar = load_calendar(path; covered = 2027:2027, sources = fixture)
            @test !is_trading_day(calendar, Date(2027, 1, 26))
            @test is_trading_day(calendar, Date(2027, 1, 27))
            @test length(calendar.holidays) == 2

            write(path, "not-a-date\n")
            @test_throws ArgumentError load_calendar(
                path; covered = 2027:2027, sources = fixture,
            )
            @test_throws ArgumentError load_calendar(
                joinpath(dir, "absent.txt"); covered = 2027:2027, sources = fixture,
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

@testset "calendar coverage and provenance" begin
    calendar = nse_calendar()

    @testset "an ordinary trading day, a weekend and a holiday" begin
        @test is_trading_day(calendar, Date(2026, 6, 3))        # a Wednesday, nothing on
        @test is_full_session(calendar, Date(2026, 6, 3))
        @test !is_trading_day(calendar, Date(2026, 6, 6))       # Saturday
        @test !is_trading_day(calendar, Date(2026, 6, 7))       # Sunday
        @test !is_trading_day(calendar, Date(2026, 6, 26))      # Moharram
        @test is_trading_day(calendar, Date(2026, 6, 25))
        @test is_trading_day(calendar, Date(2026, 6, 29))
    end

    @testset "the year boundary is handled at both ends of the coverage" begin
        # 2025-01-01 is a Wednesday and not on the holiday list, so it trades.
        @test is_trading_day(calendar, Date(2025, 1, 1))
        @test !is_trading_day(calendar, Date(2026, 12, 25))     # Christmas, a Friday
        @test is_trading_day(calendar, Date(2026, 12, 31))      # a Thursday, nothing on
        # One day past the coverage in either direction is refused, not guessed.
        @test_throws CalendarCoverageError is_trading_day(calendar, Date(2024, 12, 31))
        @test_throws CalendarCoverageError is_trading_day(calendar, Date(2027, 1, 1))
    end

    @testset "an unsupported year says how to supply it" begin
        # The message has to be actionable. An operator hitting this in a year's time needs
        # to know that the absence is deliberate and what closes it.
        message = try
            is_trading_day(calendar, Date(2027, 3, 1))
            ""
        catch error
            sprint(showerror, error)
        end
        @test occursin("2027", message)
        @test occursin("load_calendar", message)
        @test occursin("two independent sources", message)
        @test occursin("published", message)
    end

    @testset "the shipped years carry their provenance" begin
        # A holiday list is exchange data and its only claim to being right is who published
        # it. A calendar that cannot say that is a list of dates somebody typed.
        @test length(calendar.sources) >= 2
        @test all(!isempty, calendar.sources)
        @test_throws ArgumentError TradingCalendar(
            [Date(2027, 1, 1)]; covered = 2027:2027, sources = String[],
        )
        @test occursin("sources", sprint(show, calendar))
    end

    @testset "a supplied year covers a leap year correctly" begin
        # 2028 is a leap year and the shipped calendar does not reach it. Supplied, the
        # arithmetic has to handle 29 February like any other date.
        mktempdir() do dir
            path = joinpath(dir, "2028.txt")
            write(
                path,
                """
                # placeholder dates for a leap year, not an exchange list
                2028-01-26
                2028-02-29
                2028-12-25
                """,
            )
            supplied = load_calendar(
                path; covered = 2028:2028, sources = ["a test fixture, not an exchange"],
            )
            @test Date(2028, 2, 29) in supplied.holidays
            @test !is_trading_day(supplied, Date(2028, 2, 29))   # a Tuesday, listed closed
            @test is_trading_day(supplied, Date(2028, 2, 28))    # the Monday before
            @test is_trading_day(supplied, Date(2028, 3, 1))     # the Wednesday after
            @test daysinmonth(Date(2028, 2, 1)) == 29
            @test !is_trading_day(supplied, Date(2028, 1, 26))
            @test length(supplied.sources) == 1
        end
    end

    @testset "a supplied calendar must say where it came from" begin
        mktempdir() do dir
            path = joinpath(dir, "2027.txt")
            write(path, "2027-01-26\n")
            # Omitting it is a missing keyword, not a defaulted one: there is no sensible
            # default for where a holiday list came from.
            @test_throws UndefKeywordError load_calendar(path; covered = 2027:2027)
            @test_throws ArgumentError load_calendar(
                path; covered = 2027:2027, sources = String[],
            )
            supplied = load_calendar(
                path; covered = 2027:2027, sources = ["nseindia.com circular"],
            )
            @test !is_trading_day(supplied, Date(2027, 1, 26))
        end
    end

    @testset "a special session is represented and is not a full session" begin
        for date in NSE_MUHURAT
            @test is_trading_day(calendar, date)
            @test !is_full_session(calendar, date)
        end
        supplied = TradingCalendar(
            [Date(2027, 11, 5)], [Date(2027, 11, 5)];
            covered = 2027:2027, sources = ["a test fixture"],
        )
        @test is_trading_day(supplied, Date(2027, 11, 5))
        @test !is_full_session(supplied, Date(2027, 11, 5))
    end
end
